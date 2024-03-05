from PIL import Image

img = Image.open("test8.png")
imgGray = img.convert("L")
imgGray.save("test8_gray.png")