import matplotlib.pyplot as pyplot

# Plotting images from pred image array from images and probabilities
def plotImages(images_arr, probabilities = False):
    fig, axes = pyplot.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
        ax.imshow(img)
        ax.axis('off')
        print(probability[0], 'cat', probability[1], 'dog')
        if probability[0] > 0.5:
            ax.set_title("%.2f" % (probability[0]*100) + "% cat")
        else:
            ax.set_title("%.2f" % ((1-probability[0])*100) + "% dog")
    pyplot.show()