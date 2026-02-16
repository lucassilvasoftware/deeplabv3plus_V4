"""
Ponto de entrada: carrega a configuração, chama o pipeline de treino (splits fixos, processed_dataset)
e, ao final, avalia no test set da APA. Mínima lógica além da orquestração.
"""
import sys
import traceback
import torch
from config import Config
from train import train_model, run_final_evaluation_apa

if __name__ == "__main__":
    try:
        print("PyTorch version:", torch.__version__)
        print("CUDA available?", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        config = Config()
        print("\n===== Treinamento (processed_dataset, splits fixos) =====")
        train_model(config)
        print("\n===== Avaliação final no test set da APA =====")
        run_final_evaluation_apa(config)
        print("\nOK. Programa finalizado com sucesso.")
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário (Ctrl+C).", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print("\n\n" + "=" * 60, file=sys.stderr)
        print("  ERRO", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
