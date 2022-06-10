from random import randint
import math
import os
import matplotlib.pyplot as plt


def getExamples(*arg):
    args = list(arg)
    examples = args.pop(0)
    output = []
    for i in range(examples):
        element = []
        index = randint(0, args[0].shape[0]-1)
        for f in range(len(args)):
            element.append(args[f][index])
        output.append(element)
    return output


def plotExamples(path, imageList, labelList):
    print(len(imageList))
    # fig = plt.figure(figsize = (14,6))
    nrows = math.ceil(float(len(imageList[0])/3))
    # nrows = 2
    ncols = 3
    index = 1
    plot = 1
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    for i in range(len(imageList)):
        fig = plt.figure(figsize=(14, 6))
        index = 1
        for id, f in enumerate(imageList[i]):
            # print(nrows,ncols,index,len(imageList[0]))
            plt.subplot(nrows*100 + ncols*10 + index)
            # nrows+=1
            # ncols+=1
            index += 1
            plt.imshow(f, cmap="gray")
            # plt.title(f'{nrows*100 + ncols*10 + index} image')
            plt.title(labelList[id])
            # fig.canvas.draw()
            # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # append to some list this current plot
            # reset the plot/delete the plot completely

        fig.savefig(f"{path}/file{plot}.png")   # save the figure to file
        print(f"{path}/file{plot}.png")
        plot += 1
        plt.close(fig)
    # plt.tight_layout()
    # plt.show()

# plotExamples(var,labelList)


def rangeof(data):
    return (data.min(), data.max())
