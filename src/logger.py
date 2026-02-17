"""
Sistema de logs em arquivo com data e hora em cada linha.
Escreve em outputs/training/ com nome por execução (training_YYYY-MM-DD_HH-MM-SS.log)
e também imprime no console. Uso: init_logging(config) no main; depois logging.info() em todo o código.
"""
import logging
from datetime import datetime
from pathlib import Path


# Formato de cada linha: data e hora + nível + mensagem
LOG_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
LOG_LINE_FMT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_FILE_PREFIX = "training"


def init_logging(config):
    """
    Configura o logger raiz: arquivo em OUTPUTS_TRAINING com nome contendo data/hora da execução
    e saída no console. Chame uma vez no início (ex.: main.py após ensure_dirs).
    Retorna o path do arquivo de log criado.
    """
    config.ensure_dirs()
    log_dir = Path(config.OUTPUTS_TRAINING)
    now = datetime.now()
    log_filename = f"{LOG_FILE_PREFIX}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = log_dir / log_filename

    formatter = logging.Formatter(LOG_LINE_FMT, datefmt=LOG_DATETIME_FMT)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Remove handlers antigos para não duplicar ao reimportar
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return log_path
