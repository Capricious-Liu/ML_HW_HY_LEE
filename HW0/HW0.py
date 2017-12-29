from PIL import Image

def imageWeaken():
    im = Image.open("1.jpg")
    arr = im.load()
    width = im.size[0]
    height = im.size[1]

    for h in range(0, height):
        for w in range(0, width):
            arr[w,h] = tuple([int(x/2) for x in arr[w,h]])

    im.save("2.jpg")


if __name__=="__main__":
    imageWeaken()