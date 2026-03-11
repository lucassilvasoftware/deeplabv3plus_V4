"""
Converte cor de pixel em labels RGB (ex.: padronizar paleta EpaPetroBR/Bizotto).

Uso:
  python extra/label_change_pixel_color.py --label_dir=processed_dataset/1_bizotto_icmbio_3cm/label --label_output_dir=processed_dataset/1_bizotto_icmbio_3cm/label_padronizado --ori_color="(115, 76, 0)" --new_color="(114, 75, 0)"

Padronização Bizotto: (115,76,0) -> (114,75,0) para alinhar Exposed ao paper.
Saída: sempre grava cada arquivo (modificado ou cópia) para que label_output_dir fique completo.
"""
import argparse
import os
import re
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np


def _parse_color(s: str) -> tuple:
    """Converte string '(R, G, B)' ou 'R,G,B' em tupla (r, g, b) int."""
    s = s.strip()
    # Remove parênteses e extrai números
    numbers = re.findall(r"\d+", s)
    if len(numbers) != 3:
        raise ValueError("Cor deve ter 3 valores (R,G,B): {}".format(s))
    return (int(numbers[0]), int(numbers[1]), int(numbers[2]))


def _is_png(filepath: str) -> bool:
    """Considera PNG por extensão; opcionalmente usa python-magic se instalado."""
    if Path(filepath).suffix.lower() == ".png":
        return True
    try:
        import magic
        return magic.from_file(filepath, mime=True) == "image/png"
    except Exception:
        return False


class LabelChangePixelColor:
    def __init__(self, label_dir: str, label_output_dir: str, overwrite: bool = False) -> None:
        if not os.path.exists(label_dir):
            raise FileNotFoundError("{} não existe".format(label_dir))
        self.label_dir = label_dir
        self.label_output_dir = label_output_dir
        Path(label_output_dir).mkdir(parents=True, exist_ok=True)
        if not overwrite and os.listdir(label_output_dir):
            raise ValueError("{} não está vazio. Use --overwrite para permitir.".format(label_output_dir))

    def run(self, ori_color: tuple, new_color: tuple) -> None:
        total = self._count_total()
        progress = tqdm(total=total, desc="Labels")
        for root, _, files in os.walk(self.label_dir):
            rel_root = os.path.relpath(root, self.label_dir)
            out_sub = os.path.join(self.label_output_dir, rel_root) if rel_root != "." else self.label_output_dir
            for name in files:
                file_path = os.path.join(root, name)
                if not _is_png(file_path):
                    continue
                input_path = os.path.splitext(file_path)[0] + ".png"
                if not os.path.exists(input_path):
                    continue
                output_path = os.path.join(out_sub, name)
                Path(out_sub).mkdir(parents=True, exist_ok=True)

                image = Image.open(input_path)
                image = image.convert("RGB")
                pixels = image.load()
                can_save = False
                for i in range(image.size[0]):
                    for j in range(image.size[1]):
                        if pixels[i, j] == ori_color:
                            can_save = True
                            pixels[i, j] = new_color
                if can_save:
                    image.save(output_path)
                else:
                    image.save(output_path)  # cópia para manter diretório completo
                progress.update(1)
        progress.close()

    def _count_total(self) -> int:
        total = 0
        for root, _, files in os.walk(self.label_dir):
            for name in files:
                if _is_png(os.path.join(root, name)):
                    total += 1
        return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte cor de pixel em múltiplas imagens de label (RGB)."
    )
    parser.add_argument("--label_dir", type=str, required=True, help="Diretório com imagens de label")
    parser.add_argument(
        "--label_output_dir",
        type=str,
        default=None,
        help="Diretório de saída (default: label_dir + '_padronizado')",
    )
    parser.add_argument(
        "--ori_color",
        type=str,
        default="(115, 76, 0)",
        help="Cor original (ex.: '(115, 76, 0)' ou '115,76,0')",
    )
    parser.add_argument(
        "--new_color",
        type=str,
        default="(114, 75, 0)",
        help="Nova cor (ex.: '(114, 75, 0)' - Exposed paleta paper)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permitir saída em diretório não vazio",
    )
    args = parser.parse_args()

    out_dir = args.label_output_dir
    if out_dir is None:
        out_dir = args.label_dir.rstrip("/\\") + "_padronizado"

    ori = _parse_color(args.ori_color)
    new = _parse_color(args.new_color)
    LabelChangePixelColor(args.label_dir, out_dir, overwrite=args.overwrite).run(ori, new)
    print("Concluído. Saída em:", out_dir)
