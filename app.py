import os
import tkinter as tk
import customtkinter as ctk
import numpy as np
import pydicom
from PIL import Image
from PIL import ImageTk

import cv2
import matplotlib.pyplot as plt

from tkinter import filedialog


# ======================================================
# CONFIGURACIÓN
# ======================================================

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

carpeta_actual = None
dicom_files = []
indice_actual = 0
editor_roi_abierto = False
imagen_ctk = None
# Herramienta activa del ROI Editor
modo_herramienta = "rectangulo"
canvas = None

# ROI temporal
x_inicio = None
y_inicio = None
roi_temporal = None
bounding_box = None

# ROI del corte actualmente seleccionado
bounding_box_activo = None

# Diccionario de ROI por corte
bounding_boxes = {}


# ======================================================
# Estado del corte actual
# ======================================================


dataset_actual = None
imagen_actual = None
ruta_actual = None
indice_actual = 0

roi_inicio = None
roi_fin = None
dibujando_roi = False

# ======================================================
# FUNCIONES AUXILIARES
# ======================================================

def looks_like_dicom(path):
    """
    Verifica rápidamente si un archivo posee la firma DICOM.
    """

    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False

def buscar_dicom(carpeta):
    """
    Devuelve una lista de archivos DICOM ordenados por InstanceNumber.
    """

    estudios = []

    for archivo in os.listdir(carpeta):

        ruta = os.path.join(carpeta, archivo)

        if not looks_like_dicom(ruta):
            continue

        try:
            ds = pydicom.dcmread(ruta, stop_before_pixels=True)

            instance = getattr(ds, "InstanceNumber", 0)

            estudios.append((instance, ruta))

        except Exception:
            pass

    estudios.sort(key=lambda x: x[0])

    dicom_files = [ruta for _, ruta in estudios]

    return dicom_files


def buscar_primer_dicom(carpeta):
    """
    Devuelve la ruta del primer DICOM válido encontrado.
    """

    archivos = sorted(os.listdir(carpeta))

    for archivo in archivos:

        ruta = os.path.join(carpeta, archivo)

        if looks_like_dicom(ruta):
            return ruta

    return None


# ======================================================
# FUNCIONES DICOM
# ======================================================

def read_dicom_image(path):
    """
    Lee un DICOM y devuelve:
    dataset, imagen (NumPy)
    """

    ds = pydicom.dcmread(path, force=True)

    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()

    if not hasattr(ds.file_meta, "TransferSyntaxUID"):
        ds.file_meta.TransferSyntaxUID = (
            pydicom.uid.ImplicitVRLittleEndian
        )

    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    img = img * slope + intercept

    return ds, img


# ======================================================
# FUNCIONES DEL VISOR DICOM
# ======================================================

def obtener_imagen(img):
    """
    Convierte una imagen NumPy en una imagen PIL.
    """

    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)

    img = np.clip(img, vmin, vmax)

    img = (img - vmin) / (vmax - vmin)

    img = (img * 255).astype(np.uint8)

    return Image.fromarray(img)


def mostrar_imagen(imagen_pil):
    """
    Muestra una imagen PIL ajustándola automáticamente
    al tamaño del visor sin deformarla.
    """

    global imagen_ctk

    # Asegura que el frame ya conozca su tamaño
    preview.update_idletasks()

    ancho_preview = preview.winfo_width()
    alto_preview = preview.winfo_height()

    # Margen interior
    ancho_preview -= 20
    alto_preview -= 20

    img_w, img_h = imagen_pil.size
    print("Preview:", ancho_preview, "x", alto_preview)
    print("Imagen :", img_w, "x", img_h)
    # Factor de escala conservando la relación de aspecto
    escala = min(
        ancho_preview / img_w,
        alto_preview / img_h
    )

    nuevo_ancho = int(img_w * escala)
    nuevo_alto = int(img_h * escala)

    imagen_redimensionada = imagen_pil.resize(
        (nuevo_ancho, nuevo_alto),
        Image.Resampling.LANCZOS
    )

    imagen_ctk = ctk.CTkImage(
        light_image=imagen_redimensionada,
        dark_image=imagen_redimensionada,
        size=(nuevo_ancho, nuevo_alto)
    )

    imagen_label.configure(
        image=imagen_ctk,
        text=""
    )

    imagen_label.image = imagen_ctk

def mostrar_corte(indice):
    """
    Carga y muestra el corte indicado por su índice.
    """

    global dicom_files
    global indice_actual

    global dataset_actual
    global imagen_actual
    global ruta_actual
    global bounding_box_activo

    if not dicom_files:
        return

    indice_actual = indice

    ruta = dicom_files[indice]

    ds, img = read_dicom_image(ruta)

    dataset_actual = ds
    imagen_actual = img
    # imagen_actual = imagen

    ruta_actual = ruta

    imagen = obtener_imagen(img)

    mostrar_imagen(imagen)

    # ----------------------------------------
    # Restaurar ROI del corte
    # ----------------------------------------

    if indice in bounding_boxes:

        bounding_box_activo = bounding_boxes[indice].copy()

        print("ROI recuperada:")
        print(bounding_box_activo)

    else:

        bounding_box_activo = None
    
def ir_a_corte(indice):
    """
    Cambia al corte indicado y actualiza el visor.
    """
    global indice_actual

    if not dicom_files:
        return

    indice = max(0, min(indice, len(dicom_files) - 1))

    indice_actual = indice

    mostrar_corte(indice)

    slider_cortes.set(indice)

    etiqueta_corte.configure(
        text=f"Corte {indice + 1} / {len(dicom_files)}"
    )
    # ----------------------------------------
    # Cambio de corte
    # ----------------------------------------

    print(f"Corte actual: {indice}")

def cambiar_corte(valor):
    """
    Callback del slider de navegación.
    """

    indice = int(float(valor))

    ir_a_corte(indice)

def navegar_teclado(event):
    """
    Navegación mediante las flechas del teclado.
    """

    if editor_roi_abierto:
        return

    if event.keysym == "Right":
        ir_a_corte(indice_actual + 1)

    elif event.keysym == "Left":
        ir_a_corte(indice_actual - 1)

def navegar_mouse(event):
    """
    Navegación mediante la rueda del ratón.
    """

    if editor_roi_abierto:
        return

    if event.delta > 0:
        ir_a_corte(indice_actual - 1)

    elif event.delta < 0:
        ir_a_corte(indice_actual + 1)

# ======================================================
# ALGORITMO CIENTÍFICO
# ======================================================


def segment_body_outer_contour(
    img,
    crop=None,
    mode="otsu",
    hu_threshold=-500,
    min_area_pix=1000,
    show=True
):
    """
    Segmenta el contorno externo del animal/paciente y mide:
    - diámetro lateral en pixeles
    - diámetro AP en pixeles
    - área corporal en pixeles cuadrados

    crop:
        Tupla (x1, y1, x2, y2) que define la región de interés (ROI).
        Si es None, se procesa la imagen completa.
        
    mode:
    - "otsu": recomendado para capturas o imágenes sin HU confiables.
    - "hu": recomendado si los valores son HU reales.
    """

    if crop is not None:
        x1, y1, x2, y2 = crop
        roi = img[y1:y2, x1:x2]
    else:
        x1, y1 = 0, 0
        roi = img.copy()

    if mode == "hu":
        mask = roi > hu_threshold
        mask = mask.astype(np.uint8) * 255

    elif mode == "otsu":
        roi_norm = roi.astype(np.float32)
        roi_norm = roi_norm - np.nanmin(roi_norm)

        if np.nanmax(roi_norm) > 0:
            roi_norm = roi_norm / np.nanmax(roi_norm)

        roi_uint8 = (roi_norm * 255).astype(np.uint8)

        blur = cv2.GaussianBlur(roi_uint8, (7, 7), 0)

        _, mask = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Si la máscara queda invertida, se corrige.
        # Queremos que el animal sea el objeto blanco principal.
        if np.sum(mask == 255) > np.sum(mask == 0):
            mask = cv2.bitwise_not(mask)

    else:
        raise ValueError("mode debe ser 'otsu' o 'hu'.")

    kernel = np.ones((7, 7), np.uint8)

    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

   # print("Contornos encontrados:", len(contours))

    if len(contours) == 0:
        print("No se encontró ningún contorno.")
        return None

    areas = [cv2.contourArea(c) for c in contours]
    # print("Áreas:", areas)

    # Filtrar contornos pequeños
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_pix]

    print("Contornos después del filtro:", len(contours))

    if len(contours) == 0:
        # print("Todos los contornos fueron descartados.")
        return None

    # El animal debería ser el objeto de mayor área
    c = max(contours, key=cv2.contourArea)

    # Máscara rellena del contorno principal
    body_mask_roi = np.zeros_like(mask_clean)
    cv2.drawContours(body_mask_roi, [c], -1, 255, thickness=-1)

    x, y, w, h = cv2.boundingRect(c)

    area_pix2 = np.sum(body_mask_roi > 0)

    # ==========================================
    # Máscara en coordenadas globales
    # ==========================================

    mask_global = np.zeros_like(img, dtype=np.uint8)

    mask_global[
        y1:y2,
        x1:x2
    ] = body_mask_roi

    # Coordenadas globales
    x_global = x + x1
    y_global = y + y1

    # Contorno en coordenadas globales
    c_global = c.copy()
    c_global[:, 0, 0] += x1
    c_global[:, 0, 1] += y1

    D_LAT_capture_pix = w
    D_AP_capture_pix = h

    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))

        rect = plt.Rectangle(
            (x_global, y_global),
            w,
            h,
            fill=False,
            linewidth=2,
            edgecolor="red"
        )
        ax.add_patch(rect)

        

        ax.plot(c_global[:, 0, 0], c_global[:, 0, 1], linewidth=2)

        ax.set_title(
            f"LAT={D_LAT_capture_pix:.1f} pix, AP={D_AP_capture_pix:.1f} pix"
        )
        ax.set_xlabel("x [pixeles]")
        ax.set_ylabel("y [pixeles]")
        ax.grid()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(body_mask_roi, cmap="gray")
        plt.title("Máscara corporal detectada")
        plt.axis("off")
        plt.show()

    return {
        "D_LAT_capture_pix": D_LAT_capture_pix,
        "D_AP_capture_pix": D_AP_capture_pix,
        "area_capture_pix2": area_pix2,
        "bbox_x": x_global,
        "bbox_y": y_global,
        "bbox_w": w,
        "bbox_h": h,


        "contour": c_global,
        "mask": mask_global
    }


def ejecutar_segmentacion():
    """
    Ejecuta la segmentación sobre el corte actualmente mostrado.
    """

    # ----------------------------------------
    # ¿Existe una ROI manual?
    # ----------------------------------------

    if bounding_box_activo is not None:
        print("Usando ROI manual:")
        print(bounding_box_activo)

    if imagen_actual is None:
        print("No hay imagen cargada.")
        return

    # print("Shape:", imagen_actual.shape)
    # print("Tipo:", imagen_actual.dtype)
    # print("Rango:", np.min(imagen_actual), np.max(imagen_actual))

    # ----------------------------------------
    # ROI a utilizar
    # ----------------------------------------

    crop_roi = (30, 30, 560, 520)

    if bounding_box_activo is not None:

        crop_roi = (
            min(bounding_box_activo["x1"], bounding_box_activo["x2"]),
            min(bounding_box_activo["y1"], bounding_box_activo["y2"]),
            max(bounding_box_activo["x1"], bounding_box_activo["x2"]),
            max(bounding_box_activo["y1"], bounding_box_activo["y2"])
        )

        print("Usando ROI manual:", crop_roi)

    resultado = segment_body_outer_contour(
        imagen_actual,
        crop=crop_roi,
        mode="otsu",
        show=False
    )

    if resultado is not None:
        print(
            f"Área = {resultado['area_capture_pix2']} pix² | "
            f"LAT = {resultado['D_LAT_capture_pix']} px | "
            f"AP = {resultado['D_AP_capture_pix']} px"
        )
    
        mostrar_segmentacion(resultado)

        boton_aceptar.pack(
            fill="x",
            padx=10,
            pady=(5, 5)
        )

        boton_roi_manual.pack(
            fill="x",
            padx=10,
            pady=(5, 10)
    )

    boton_aceptar.configure(state="normal")
    boton_roi_manual.configure(state="normal")

    boton_segmentar.configure(
        text="Segmentación realizada",
        state="disabled"
    )

def mostrar_segmentacion(resultado):
    """
    Dibuja el resultado de la segmentación sobre la imagen.
    """
    global bounding_box_activo

    img = imagen_actual.copy()

    img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # ----------------------------------------
    # Bounding Box definido por el usuario
    # ----------------------------------------

    if bounding_box_activo is not None:

        x1 = min(
            bounding_box_activo["x1"],
            bounding_box_activo["x2"]
        )

        x2 = max(
            bounding_box_activo["x1"],
            bounding_box_activo["x2"]
        )

        y1 = min(
            bounding_box_activo["y1"],
            bounding_box_activo["y2"]
        )

        y2 = max(
            bounding_box_activo["y1"],
            bounding_box_activo["y2"]
        )

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    cv2.drawContours(
        img,
        [resultado["contour"]],
        -1,
        (0, 255, 255),
        2
    )

    imagen = Image.fromarray(img)

    mostrar_imagen(imagen)
# ======================================================
# FUNCIONES DE LA INTERFAZ
# ======================================================

def reiniciar_interfaz():
    """
    Restaura la interfaz al estado inicial.
    """

    boton_aceptar.pack_forget()
    boton_roi_manual.pack_forget()

    boton_segmentar.configure(
        text="Continuar",
        state="disabled"
    )



def seleccionar_carpeta():

    global carpeta_actual, dicom_files
    
    reiniciar_interfaz()

    carpeta =filedialog.askdirectory(
        title="Seleccionar caprta con archivos DICOM"
    )

    if not carpeta:
        return

    

    carpeta_actual = carpeta

    nombre_estudio = os.path.basename(carpeta)

    archivos = sorted(os.listdir(carpeta))

    dicom_validos = sum(
        looks_like_dicom(os.path.join(carpeta, f))
        for f in archivos
    )

    etiqueta_info.configure(
    text=(
        f"📂 Estudio cargado\n\n"
        f"{nombre_estudio}\n\n"
        f"📄 {len(archivos)} archivos encontrados\n"
        f"✔ {dicom_validos} archivos DICOM válidos"
    )
)

    estado_label.configure(
        text="Estado: Cargando vista previa..."
    )

    dicom_files = buscar_dicom(carpeta)

    if len(dicom_files) == 0:

        estado_label.configure(
            text="Estado: No se encontraron archivos DICOM."
        )

        return

    slider_cortes.configure(
        from_=0,
        to=len(dicom_files) - 1,
        number_of_steps=max(len(dicom_files) - 1, 1),
    )

    slider_cortes.set(0)

    mostrar_corte(0)

    # Habilitar el botón nuevamente
    boton_segmentar.configure(
        text="Continuar",
        state="normal"
    )

    print("Estado del botón:", boton_segmentar.cget("state"))

    estado_label.configure(
        text="Estado: Vista previa cargada correctamente."
    )
        
    estado_label.configure(
        text="Estado: Vista previa cargada correctamente."
    ) 

    boton_segmentar.configure(
            text="Continuar",
            state="normal"
    )

    etiqueta_corte.grid(
            row=3,
            column=0,
            pady=(0, 10)
        )

def click_roi(event):
    """
    Guarda el punto inicial de la ROI.
    """

    global roi_inicio
    global dibujando_roi

    roi_inicio = (event.x, event.y)
    dibujando_roi = True

    print("Inicio ROI:", roi_inicio)

def soltar_roi(event):
    """
    Guarda el punto final de la ROI.
    """

    global roi_fin
    global dibujando_roi

    roi_fin = (event.x, event.y)
    dibujando_roi = False

    print("Fin ROI:", roi_fin)


def abrir_editor_roi():

    """
    Abre la ventana del editor de ROI.
    """
    global canvas
    global editor_roi_abierto

    ventana_roi = ctk.CTkToplevel(app)
    
    editor_roi_abierto = True

    ventana_roi.title("ROI Editor")
    ventana_roi.geometry("850x700")

    ventana_roi.transient(app)
    ventana_roi.grab_set()
    ventana_roi.focus_force()

    def cerrar_editor():
        global editor_roi_abierto

        editor_roi_abierto = False
        ventana_roi.destroy()


    ventana_roi.protocol(
        "WM_DELETE_WINDOW",
        cerrar_editor
    )

    # ==========================
    # Barra de herramientas
    # ==========================

    toolbar = ctk.CTkFrame(
        ventana_roi,
        height=45
    )

    toolbar.pack_propagate(False)

    toolbar.pack(
        fill="x",
        padx=10,
        pady=(10, 5)
    )

    # ==========================
    # Herramientas
    # ==========================

    boton_rectangulo = ctk.CTkButton(
        toolbar,
        text="▭ Rectángulo",
        width=120,
        command=lambda: seleccionar_herramienta("rectangulo")
    )

    boton_rectangulo.pack(
        side="left",
        padx=8,
        pady=6
    )

    boton_circulo = ctk.CTkButton(
    toolbar,
    text="◯ Círculo",
    width=120,
    command=lambda: seleccionar_herramienta("circulo")
)

    boton_circulo.pack(
        side="left",
        padx=8,
        pady=6
    )

    boton_libre = ctk.CTkButton(
    toolbar,
    text="✏ Mano alzada",
    width=140,
    command=lambda: seleccionar_herramienta("libre")
)

    boton_libre.pack(
        side="left",
        padx=8,
        pady=6
    )
    # ==========================
    # Área del Canvas
    # ==========================

    canvas_frame = ctk.CTkFrame(ventana_roi)

    canvas_frame.pack(
        fill="both",
        expand=True,
        padx=10,
        pady=(0, 10)
    )

    canvas = tk.Canvas(
        canvas_frame,
        bg="black",
        cursor="cross"
    )

    canvas.pack(
        fill="both",
        expand=True
    )

    canvas.bind("<Button-1>", iniciar_dibujo)
    canvas.bind("<B1-Motion>", actualizar_dibujo)
    canvas.bind("<ButtonRelease-1>", finalizar_dibujo)

    # Mostrar la imagen actual
    if imagen_actual is not None:

        imagen = Image.fromarray(
            imagen_actual.astype(np.uint8)
        )

        # Ajustar la imagen al tamaño del editor
        imagen.thumbnail((900, 550))

        foto = ImageTk.PhotoImage(imagen)

        canvas.create_image(
            0,
            0,
            anchor="nw",
            image=foto
        )

        # Mantener una referencia para que la imagen no desaparezca
        canvas.image = foto




    # ==========================
    # Barra inferior
    # ==========================

    bottom = ctk.CTkFrame(ventana_roi)

    bottom.pack(
        fill="x",
        padx=10,
        pady=(0, 10)
    )

    ctk.CTkButton(
        bottom,
        text="Cancelar",
        command=cerrar_editor
    ).pack(
        side="left",
        padx=10,
        pady=10
    )

    ctk.CTkButton(
    bottom,
    text="Aplicar ROI",
    command=lambda: aplicar_roi(ventana_roi)
).pack(
    side="right",
    padx=10,
    pady=10
)


# ======================================================
# VENTANA PRINCIPAL
# ======================================================
app = ctk.CTk()

app.bind("<Right>", navegar_teclado)
app.bind("<Left>", navegar_teclado)
app.bind("<MouseWheel>", navegar_mouse)

app.title("VetCT Geometry")

app.geometry("1000x650")
app.minsize(900, 600)

app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(1, weight=1)
def seleccionar_herramienta(herramienta):
    """
    Cambia la herramienta activa del editor ROI.
    """

    global modo_herramienta

    modo_herramienta = herramienta

    print(f"Herramienta activa: {modo_herramienta}")
def iniciar_dibujo(event):
    """
    Inicia el dibujo de la ROI.
    """

    global x_inicio, y_inicio

    x_inicio = event.x
    y_inicio = event.y


def actualizar_dibujo(event):
    """
    Actualiza el dibujo de la ROI mientras se mueve el mouse.
    """

    global roi_temporal

    if modo_herramienta != "rectangulo":
        return

    if x_inicio is None or y_inicio is None:
        return

    if roi_temporal is not None:
        canvas.delete(roi_temporal)

    roi_temporal = canvas.create_rectangle(
        x_inicio,
        y_inicio,
        event.x,
        event.y,
        outline="red",
        width=2
    )


def finalizar_dibujo(event):
    """
    Finaliza el dibujo de la ROI.
    """

    global bounding_box

    if modo_herramienta != "rectangulo":
        return

    bounding_box = {
        "x1": x_inicio,
        "y1": y_inicio,
        "x2": event.x,
        "y2": event.y
    }

    print(bounding_box)

def aplicar_roi(ventana):
    """
    Confirma la ROI seleccionada.
    """

    global bounding_box_activo
    global editor_roi_abierto

    if bounding_box is None:
        print("No hay ninguna ROI seleccionada.")
        return

    bounding_box_activo = bounding_box.copy()

    # Guardar la ROI del corte actual
    bounding_boxes[indice_actual] = bounding_box_activo.copy()

    print("ROI aplicada:")
    print(bounding_box_activo)

    editor_roi_abierto = False

    ventana.destroy()

    # Ejecutar nuevamente la segmentación
    ejecutar_segmentacion()
# ======================================================
# HEADER
# ======================================================

header = ctk.CTkFrame(
    app,
    fg_color="transparent"
)

header.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=20,
    pady=20
)

titulo = ctk.CTkLabel(
    header,
    text="VetCT Geometry",
    font=("Arial", 34, "bold")
)

titulo.pack()

subtitulo = ctk.CTkLabel(
    header,
    text="Estimación volumétrica mediante Tomografía Computarizada",
    font=("Arial", 17)
)

subtitulo.pack(pady=(5, 0))

# ======================================================
# PANEL PRINCIPAL
# ======================================================

main = ctk.CTkFrame(app)

main.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=20,
    pady=10
)

main.grid_rowconfigure(0, weight=1)
main.grid_columnconfigure(0, weight=1)
main.grid_columnconfigure(1, weight=2)

# ======================================================
# PANEL IZQUIERDO
# ======================================================

left = ctk.CTkFrame(main)

left.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(10, 5),
    pady=10
)

titulo_estudio = ctk.CTkLabel(
    left,
    text="ESTUDIO",
    font=("Arial", 18, "bold")
)

titulo_estudio.pack(pady=(20, 15))

boton = ctk.CTkButton(
    left,
    text="📂 Seleccionar estudio",
    width=220,
    command=seleccionar_carpeta
)

boton.pack(pady=15)

etiqueta_info = ctk.CTkLabel(
    left,
    text="Ningún estudio cargado.",
    font=("Arial", 14),
    justify="center",
    wraplength=250
)

etiqueta_info.pack(pady=20)


# ======================================================
# HERRAMIENTAS
# ======================================================

herramientas_frame = ctk.CTkFrame(left)

herramientas_frame.pack(
    fill="x",
    padx=15,
    pady=(10, 15)
)

boton_segmentar = ctk.CTkButton(
    herramientas_frame,
    text="Continuar",
    state="disabled",
    command=ejecutar_segmentacion
)

boton_segmentar.pack(
    fill="x",
    padx=10,
    pady=10
)


boton_aceptar = ctk.CTkButton(
    herramientas_frame,
    text="✔ Usar segmentación",
    state="disabled"
)

boton_roi_manual = ctk.CTkButton(
    herramientas_frame,
    text="✏ Dibujar ROI manual",
    state="disabled",
    command=abrir_editor_roi
)


# ======================================================
# PANEL DERECHO
# ======================================================

right = ctk.CTkFrame(main)

right.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(5, 10),
    pady=10
)

# Configuración del grid del panel derecho
right.grid_columnconfigure(0, weight=1)

right.grid_rowconfigure(0, weight=0)   # Título
right.grid_rowconfigure(1, weight=1)   # Visor (crece)
right.grid_rowconfigure(2, weight=0)   # Slider
right.grid_rowconfigure(3, weight=0)   # Información del corte


# ------------------------------
# TÍTULO
# ------------------------------

titulo_preview = ctk.CTkLabel(
    right,
    text="VISTA PREVIA",
    font=("Arial", 18, "bold")
)

titulo_preview.grid(
    row=0,
    column=0,
    pady=(20, 10)
)


# ------------------------------
# VISOR
# ------------------------------

preview = ctk.CTkFrame(right)

preview.grid(
    row=1,
    column=0,
    padx=15,
    pady=10,
    sticky="nsew"
)

preview.grid_rowconfigure(0, weight=1)
preview.grid_columnconfigure(0, weight=1)

imagen_label = ctk.CTkLabel(
    preview,
    text="Seleccione un estudio DICOM",
    font=("Arial", 18)
)

imagen_label.grid(
    row=0,
    column=0,
    sticky="nsew"
)

imagen_label.bind("<Button-1>", click_roi)
imagen_label.bind("<ButtonRelease-1>", soltar_roi)

# ------------------------------
# SLIDER
# ------------------------------

slider_cortes = ctk.CTkSlider(
    right,
    from_=0,
    to=0,
    number_of_steps=1,
    command=cambiar_corte
)

slider_cortes.grid(
    row=2,
    column=0,
    sticky="ew",
    padx=20,
    pady=(5, 10)
)


# ------------------------------
# INFORMACIÓN DEL CORTE
# ------------------------------

etiqueta_corte = ctk.CTkLabel(
    right,
    text=""
)

etiqueta_corte.grid(
    row=3,
    column=0,
    pady=(0, 10)
)
# ======================================================
# BARRA DE ESTADO
# ======================================================

estado = ctk.CTkFrame(app)

estado.grid(
    row=2,
    column=0,
    sticky="ew",
    padx=20,
    pady=(0, 20)
)

estado_label = ctk.CTkLabel(
    estado,
    text="Estado: Esperando selección de estudio.",
    anchor="w"
)

estado_label.pack(
    fill="x",
    padx=10,
    pady=10
)

# ======================================================

app.mainloop()

