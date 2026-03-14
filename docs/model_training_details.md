# Model Training Details

Three deep learning models were used in this project:

1. InceptionV3 [Final]
2. ResNet50
3. MobileNetV2

Transfer learning was used with pretrained ImageNet weights.

Training Process:

Phase 1:
- Base model layers frozen
- Custom classification head trained

Phase 2:
- Last layers of base model unfrozen
- Fine tuning performed with lower learning rate

Training Parameters:

Image Size: 299x299
Batch Size: 32
Optimizer: Adam
Loss Function: Categorical Crossentropy
Epochs: 15
