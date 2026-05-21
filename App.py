import gradio as gr
import cv2
import pandas as pd
import os
import zipfile
from datetime import datetime
import tempfile
import shutil

# ==============================
# LÓGICA DE PROCESAMIENTO (Pura)
# ==============================

def draw_points(frame, points, current_frame_idx):
    """Dibuja los puntos sobre el frame actual."""
    img_copy = frame.copy()
    for row in points:
        if row["Frame"] == current_frame_idx:
            color = (0, 0, 255) if row["Valve"] == "Mitral" else (255, 0, 0)
            cv2.circle(img_copy, (row["X"], row["Y"]), 6, color, -1)
    return img_copy

# ==============================
# CONTROLADORES DE ESTADO
# ==============================

def load_video(video_path):
    if not video_path:
        return None, pd.DataFrame(), gr.update(maximum=0, value=0), [], [], 0, "", None

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames = []
    
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convertir BGR a RGB para la correcta visualización en web
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        return None, pd.DataFrame(), gr.update(maximum=0, value=0), [], [], 0, video_name, None

    slider_update = gr.update(minimum=0, maximum=len(frames)-1, value=0)
    
    return frames[0], pd.DataFrame(), slider_update, frames, [], 0, video_name, None

def change_frame(frame_number, frames, points):
    frame_idx = int(frame_number)
    if not frames or frame_idx >= len(frames):
        return None, frame_idx, None
    
    img = draw_points(frames[frame_idx], points, frame_idx)
    return img, frame_idx, None # Reiniciamos el punto seleccionado al cambiar de frame

def select_point(evt: gr.SelectData, frames, points, frame_idx):
    x, y = int(evt.index[0]), int(evt.index[1])
    selected_point = (x, y)

    img = draw_points(frames[frame_idx], points, frame_idx)
    # Preview del punto seleccionado en verde
    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

    return img, selected_point

def add_point(valve_type, frames, points, frame_idx, selected_point, video_name):
    if selected_point is None or not frames:
        return gr.update(), pd.DataFrame(points), points

    x, y = selected_point

    # Eliminar punto previo de la misma válvula en el mismo frame (Clean logic)
    updated_points = [
        p for p in points
        if not (p["Frame"] == frame_idx and p["Valve"] == valve_type)
    ]

    updated_points.append({
        "Video": video_name,
        "Frame": frame_idx,
        "X": x,
        "Y": y,
        "Valve": valve_type
    })

    img = draw_points(frames[frame_idx], updated_points, frame_idx)
    return img, pd.DataFrame(updated_points), updated_points

def delete_points_on_frame(frames, points, frame_idx):
    if not frames:
        return gr.update(), pd.DataFrame(points), points
        
    updated_points = [p for p in points if p["Frame"] != frame_idx]
    img = frames[frame_idx].copy()
    
    return img, pd.DataFrame(updated_points), updated_points

def download_data(frames, points):
    if not frames or not points:
        return None

    # Uso de directorio temporal para evitar saturar el servidor
    temp_dir = tempfile.mkdtemp()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join(temp_dir, f"dataset_{timestamp}")
    
    originales_folder = os.path.join(folder_name, "originales")
    mitral_folder = os.path.join(folder_name, "mitral")
    tricuspide_folder = os.path.join(folder_name, "tricuspide")

    for f in [originales_folder, mitral_folder, tricuspide_folder]:
        os.makedirs(f, exist_ok=True)

    # Guardar Excel
    df = pd.DataFrame(points)
    df.to_excel(os.path.join(folder_name, "puntos.xlsx"), index=False)

    frames_unicos = set([p["Frame"] for p in points])

    for f_idx in frames_unicos:
        # Volver a BGR para guardar con OpenCV
        base_img = cv2.cvtColor(frames[f_idx], cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(originales_folder, f"frame_{f_idx}.png"), base_img)

        mitral_img = base_img.copy()
        tricuspide_img = base_img.copy()

        for row in points:
            if row["Frame"] == f_idx:
                if row["Valve"] == "Mitral":
                    cv2.circle(mitral_img, (row["X"], row["Y"]), 6, (0, 0, 255), -1)
                elif row["Valve"] == "Tricúspide":
                    cv2.circle(tricuspide_img, (row["X"], row["Y"]), 6, (255, 0, 0), -1)

        cv2.imwrite(os.path.join(mitral_folder, f"frame_{f_idx}.png"), mitral_img)
        cv2.imwrite(os.path.join(tricuspide_folder, f"frame_{f_idx}.png"), tricuspide_img)

    zip_path = os.path.join(temp_dir, f"dataset_{timestamp}.zip")
    
    # Crear ZIP de manera limpia
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_name)

    return zip_path

# ==============================
# INTERFAZ GRADIO
# ==============================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🫀 Etiquetado de Anillo Mitral / Tricúspide")
    gr.Markdown("Sube un video, selecciona un frame, haz clic en la imagen para fijar las coordenadas y clasifica la válvula.")

    # Estados de sesión (Reemplazan a las variables globales)
    state_frames = gr.State([])
    state_points = gr.State([])
    state_frame_idx = gr.State(0)
    state_video_name = gr.State("")
    state_selected_point = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Subir Video", height=300)
            valve_selector = gr.Radio(choices=["Mitral", "Tricúspide"], label="Tipo de válvula", value="Mitral")
            
            with gr.Row():
                add_btn = gr.Button("Agregar Punto", variant="primary")
                delete_btn = gr.Button("Eliminar Puntos", variant="stop")
            
            download_btn = gr.Button("Empaquetar y Descargar Dataset")
            file_output = gr.File(label="Dataset ZIP")
            
        with gr.Column(scale=2):
            frame_slider = gr.Slider(minimum=0, maximum=1, step=1, label="Navegador de Frames", value=0)
            frame_display = gr.Image(label="Área de Etiquetado", interactive=True, height=500)
            dataframe_output = gr.Dataframe(headers=["Video", "Frame", "X", "Y", "Valve"], interactive=False)

    # Eventos y flujos de datos
    video_input.upload(
        load_video,
        inputs=[video_input],
        outputs=[frame_display, dataframe_output, frame_slider, state_frames, state_points, state_frame_idx, state_video_name, state_selected_point]
    )

    frame_slider.change(
        change_frame,
        inputs=[frame_slider, state_frames, state_points],
        outputs=[frame_display, state_frame_idx, state_selected_point]
    )

    frame_display.select(
        select_point,
        inputs=[state_frames, state_points, state_frame_idx],
        outputs=[frame_display, state_selected_point]
    )

    add_btn.click(
        add_point,
        inputs=[valve_selector, state_frames, state_points, state_frame_idx, state_selected_point, state_video_name],
        outputs=[frame_display, dataframe_output, state_points]
    )

    delete_btn.click(
        delete_points_on_frame,
        inputs=[state_frames, state_points, state_frame_idx],
        outputs=[frame_display, dataframe_output, state_points]
    )

    download_btn.click(
        download_data,
        inputs=[state_frames, state_points],
        outputs=[file_output]
    )

if __name__ == "__main__":
    demo.launch()
