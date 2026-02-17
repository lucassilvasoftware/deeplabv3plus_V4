"""
Ponto de entrada: carrega a configuração, chama o pipeline de treino (splits fixos, processed_dataset)
e, ao final, avalia no test set da APA. Logs em arquivo (data/hora) + console.
"""
import sys
import logging
import torch
from config import Config
from logger import init_logging
from train import train_model, run_final_evaluation_apa

if __name__ == "__main__":
    try:
        config = Config()
        log_path = init_logging(config)
        logging.info("Arquivo de log: %s", log_path)
        logging.info("PyTorch version: %s", torch.__version__)
        logging.info("CUDA available? %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logging.info("GPU: %s", torch.cuda.get_device_name(0))

        logging.info("===== Treinamento (processed_dataset, splits fixos) =====")
        train_model(config)
        logging.info("===== Avaliação final no test set da APA =====")
        run_final_evaluation_apa(config)
        logging.info("OK. Programa finalizado com sucesso.")
    except KeyboardInterrupt:
        logging.warning("Interrompido pelo usuário (Ctrl+C).")
        sys.exit(130)
    except Exception:
        logging.exception("ERRO durante a execução")
        sys.exit(1)
