from senior_research import preprocessing as pre

images, labels = pre.convert_img2array()
train_images, test_images, train_labels, test_labels = pre.create_train_test_sets(images, labels, 0.2)

def main():
    print("images shape ", images.shape)
    print("labels shape ", labels.shape)
    print("train_images shape ", train_images.shape)
    print("test_images shape ", test_images.shape)
    print("singe image shape ", train_images[0].shape)

if __name__ == "__main__":
    main()