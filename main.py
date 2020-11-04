from senior_research import preprocessing

images, labels = preprocessing.convert_img2array()

def main():
    print(images.shape)
    print(labels.shape)

if __name__ == "__main__":
    main()