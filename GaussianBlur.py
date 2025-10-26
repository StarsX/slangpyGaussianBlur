from app import App
import slangpy as spy
import math
import time
import sys
from pathlib import Path

def getImageDimensions(input_file):
    bmp = spy.Bitmap(input_file)
    return spy.int2(bmp.width, bmp.height)

input_file = "vangogh.jpg"

if len(sys.argv) > 1:
    input_file = sys.argv[1]

imageDim = getImageDimensions(input_file)

# Create the app and load the slang kernel.
app = App(width = imageDim.x, height = imageDim.y, title = "slangpy: Gaussian Blur")

gaussian_blur_program = app.device.load_program(module_name = "gaussianBlur.slang", entry_point_names = ["main"])
gaussian_blur_kernel = app.device.create_compute_kernel(program = gaussian_blur_program)

#copy_program = app.device.load_program(module_name = "gaussianBlur.slang", entry_point_names = ["copy"])
#copy_kernel = app.device.create_compute_kernel(program = copy_program)

data_path = Path(__file__).parent

# Load input texture.
input_map = spy.Tensor.load_from_image(
    app.device, data_path.joinpath(input_file), linearize = True
)

# Allocate a tensor for output
output = spy.Tensor.empty_like(input_map)

while app.process_events():

    blurRadius = int((math.sin(time.perf_counter() * math.pi) * 0.5 + 0.5) * 20.0)

    # Dispatch compute shader.
    gaussian_blur_kernel.dispatch(thread_count = [output.shape[1], output.shape[0], 1], BlurRadius = blurRadius, Src = input_map, Dst = output)
    #copy_kernel.dispatch(thread_count = [output.shape[1], output.shape[0], 1], Src = input_map, Dst = output)

    # Blit tensor to screen.
    app.blit(source = output, tonemap = False)

    # Present the window.
    app.present()
