set -x
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

python test_pixel_link_on_any_image.py
