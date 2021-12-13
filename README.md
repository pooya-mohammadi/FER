## FER (Facial Emotion Recognition)

In this repository we present our experiments on fer-2013 dataset


# Download Dataset
Download the official [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset, and place it in the dataset folder with the following folder structure `datasets/fer2013.csv`

# Installation
`pip install -r requirments.txt`

# Train
## VGG
`python train.py --model-name vgg --crop-size 40 --batch-size 8 --n-workers 4`

## 🤝 Team members

<table>
  <tr>
    <td align="center">
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/55460936?v=4" width="100px;" alt="Pooya Mohammadi no GitHub"/><br>
        <sub>
          <b>Pooya Mohammadi Kazaj</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/dornasabet">
        <img src="https://avatars.githubusercontent.com/u/74057278?v=4" width="100px;" alt="Dorna Sabet"/><br>
        <sub>
          <b>Dorna Sabet</b>
        </sub>
      </a>
    </td>

</table>


# References
1. https://github.com/usef-kh/fer