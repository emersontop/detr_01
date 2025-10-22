import os
import pandas as pd
import matplotlib.pyplot as plt

# Diretório base dos logs do Lightning
LIGHTNING_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lightning_logs')

# Encontra a última versão de execução
versions = [d for d in os.listdir(LIGHTNING_LOGS_DIR) if d.startswith('version_')]
if not versions:
    raise FileNotFoundError('Nenhuma versão encontrada em lightning_logs.')
latest_version = sorted(versions, key=lambda x: int(x.split('_')[1]))[-1]
metrics_path = os.path.join(LIGHTNING_LOGS_DIR, latest_version, 'metrics.csv')
print(f'Lendo métricas de: {metrics_path}')

# Lê o arquivo de métricas
metrics = pd.read_csv(metrics_path)

# Gráficos de loss lado a lado por época
train_epoch_loss = metrics.groupby('epoch')['training_loss'].mean()
val_epoch_loss = metrics.groupby('epoch')['validation/loss'].mean() if 'validation/loss' in metrics.columns else None

# Suavização com média móvel
window = 3  # você pode ajustar o tamanho da janela
train_epoch_loss_smooth = train_epoch_loss.rolling(window, min_periods=1).mean()
val_epoch_loss_smooth = val_epoch_loss.rolling(window, min_periods=1).mean() if val_epoch_loss is not None else None

epochs = train_epoch_loss.index
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# Training Loss por época
axs[0, 0].plot(epochs, train_epoch_loss, label='Training Loss (epoch)', color='blue')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_title('Training Loss por Época')
axs[0, 0].legend()
axs[0, 0].grid(True)
# Validation Loss por época
if val_epoch_loss is not None:
    axs[0, 1].plot(val_epoch_loss.index, val_epoch_loss, label='Validation Loss (epoch)', color='orange')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Validation Loss por Época')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
# Training Loss suavizado
axs[1, 0].plot(epochs, train_epoch_loss_smooth, label='Training Loss (smoothed)', color='green')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].set_title('Training Loss Suavizado')
axs[1, 0].legend()
axs[1, 0].grid(True)
# Validation Loss suavizado
if val_epoch_loss_smooth is not None:
    axs[1, 1].plot(val_epoch_loss_smooth.index, val_epoch_loss_smooth, label='Validation Loss (smoothed)', color='red')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].set_title('Validation Loss Suavizado')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
plt.tight_layout()
plt.show()