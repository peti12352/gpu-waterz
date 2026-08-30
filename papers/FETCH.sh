#!/bin/sh
# Re-get the PDFs cited in SOURCES.md. Run from papers/.
set -eu
curl -k -fsSL -o mutex_watershed_wolf2020.pdf "https://arxiv.org/pdf/1904.12654.pdf"
curl -k -fsSL -o mutex_watershed_wolf2018.pdf "https://arxiv.org/pdf/1705.08369.pdf"
curl -k -fsSL -o parallel_watershed_gpu_2024.pdf "https://arxiv.org/pdf/2410.08946.pdf"
curl -k -fsSL -o funke_structured_loss_2017.pdf "https://arxiv.org/pdf/1709.02974.pdf"
curl -k -fsSL -o nunez_iglesias_gala_2013.pdf "https://arxiv.org/pdf/1303.5942.pdf"
ls -la *.pdf
