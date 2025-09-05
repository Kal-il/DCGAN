import os

import smtplib
import subprocess
import time
import shutil  # NOVO: Para a alternativa de criar um ZIP
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase  # NOVO: Para anexos
from email import encoders  # NOVO: Para anexos

# NOVO: Importe a função que cria o GIF e o seu arquivo de configuração
import config
from utils import create_evolution_gif

# --- Configurações do E-mail ---
SENDER_EMAIL = 'kalilgarcia38@gmail.com'
SENDER_PASSWORD = 'bceo hviz qdbw lnee'
RECEIVER_EMAIL = 'kalilcanuto@gmail.com'  # Altere para o seu e-mailimport os

# --- Configurações do Servidor SMTP (exemplo para o Gmail) ---
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# --- Script a ser executado ---
SCRIPT_TO_RUN = 'train.py'

# NOVO: Define os diretórios de saída para encontrar as imagens e salvar o GIF
output_dir = os.path.join("experiments", config.EXPERIMENT_NAME)
epochs_images_dir = os.path.join(output_dir, "images")
gif_path = os.path.join(output_dir, "training_evolution.gif")


def send_notification_email(elapsed_time, attachment_path=None):  # NOVO: Adicionado attachment_path
    """
    Envia um e-mail notificando sobre o término do treinamento, com anexo opcional.
    """
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("!!! ERRO: Variáveis de ambiente de e-mail não definidas.")
        return

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"

    message = MIMEMultipart()
    message['From'] = SENDER_EMAIL
    message['To'] = RECEIVER_EMAIL

    message['Subject'] = f"Treinamento Concluído: {config.EXPERIMENT_NAME}"
    body = f"""
    O script de treinamento `{SCRIPT_TO_RUN}` foi concluído com sucesso.
    <br><br>
    <b>Tempo total de execução:</b> {time_str}
    <br><br>
    O anexo contém a evolução do treinamento.
        """

    message.attach(MIMEText(body, 'html'))

    # NOVO: Lógica para adicionar o anexo
    if attachment_path and os.path.exists(attachment_path):
        print(f"Anexando o arquivo: {attachment_path}")
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(attachment_path)}",
        )
        message.attach(part)
        print("Arquivo anexado.")

    try:
        print("Enviando e-mail de notificação...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
        print("E-mail enviado com sucesso!")
    except Exception as e:
        print(f"!!! Falha ao enviar o e-mail: {e}")


def generate_gif_and_send_email(elapsed_time):
    attachment_to_send = None
    create_evolution_gif(epochs_images_dir, gif_path)
    if os.path.exists(gif_path):
        attachment_to_send = gif_path

    send_notification_email(elapsed_time, attachment_path=attachment_to_send)
