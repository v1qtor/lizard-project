# 🦎 Lizard Species Classification — Defence Study Guide

> **How to use this guide:** Read it like a story from top to bottom. Every section matches a section in the notebook. The "❓ Possible Questions" boxes at the end of each section are what your teacher will likely ask, with full answers ready for you.

---

## 🧠 The Big Picture — What is this project actually doing?

**The Story:**
Imagine you've never seen a dog in your life, but you want to be able to recognise 7 different dog breeds just by looking at photos. You can't possibly memorise every dog from scratch — there are thousands. Instead, you find an expert dog judge (someone who has studied millions of animal photos their whole life), and you say: *"Hey, I know you know about animals in general. Can you just learn the specific differences between these 7 lizard types from me?"*

That's exactly what this project does. We take a neural network called **EfficientNetV2-S** that was already trained on millions of images, and we *teach it the specific differences between 7 lizard species* from a small dataset for a Kaggle competition.

The 7 lizard species are:
1. Black Spiny-Tailed Iguana
2. Brown Anole
3. Cuban Knight Anole
4. Desert Iguana
5. Green Anole
6. Green Iguana
7. Lesser Antillean Iguana

The **evaluation metric** is **F1 Score (Macro)** — more on that later.

---

### 🤖 The Model Choice — EfficientNetV2-S vs EfficientNetB3

The notebook gives you a choice between two models (controlled by `USE_V2 = True`):

**EfficientNetB3** (the older option):
- Part of the original EfficientNet family released by Google in 2019
- Works by scaling up a baseline network in three dimensions simultaneously: **depth** (more layers), **width** (more neurons per layer), and **resolution** (larger input images)
- The "B3" means it's the 3rd level of scaling — a balanced middle ground between speed and accuracy
- Very strong general-purpose image classifier, but designed for larger datasets

**EfficientNetV2-S** (what we use — `USE_V2 = True`):
- Released by Google in 2021 — a complete redesign, not just a bigger B3
- The "V2" means second generation; "S" means Small
- Key improvements over B3:
  - **Trains 5–11x faster** thanks to a smarter training recipe
  - Uses **Fused-MBConv** blocks in the early layers (combines operations that were separate in V1, which is faster on modern hardware like TPUs and GPUs)
  - Specifically designed to perform well on **smaller datasets** — which is exactly our situation with a limited number of lizard photos
  - Better at handling the 300×300 input size we use

**Why EfficientNetV2-S was chosen for this project:**
The lizard dataset is small (a few hundred images per class at most). Larger, more complex models tend to overfit on small datasets because they have too many parameters to learn. EfficientNetV2-S hits the sweet spot — powerful enough to extract rich visual features, but compact enough not to overfit. The expected gain over B3 on this dataset is **+2–3% F1 score**.

**How these models work — the core idea:**
Both models are built from stacked **MBConv blocks** (Mobile Inverted Bottleneck Convolution). Think of each block as a small specialist:
1. It **expands** the input to a higher-dimensional space (to see more patterns)
2. It applies a **depthwise convolution** — filtering each colour/feature channel independently (very efficient)
3. It **squeezes** the result back down (keeps only the important information)
4. It uses a **skip connection** (adds the original input back) so information is never lost

Stacked together, these blocks build a hierarchy of understanding — from simple edges in the first layers to complex lizard-specific features in the final layers. The pre-trained weights from ImageNet mean all of this learned understanding comes for free — we just need to point it at lizards.

---

## Section 1 — Setup & Configuration

### What's happening?
This is the "setting up the kitchen before cooking" section. We:
- Import all the tools (libraries) we need
- Set up the GPU (graphics card) for faster computing
- Define all the important settings/numbers that the rest of the notebook uses

### Key concepts to know:

**What is a library?**
A library is a collection of pre-written code. Instead of writing everything from scratch, we import tools others already built. Like using a calculator instead of doing maths by hand.

**TensorFlow & Keras:**
TensorFlow is Google's framework for building neural networks. Keras is the user-friendly layer on top of it — think of TensorFlow as the engine of a car, and Keras as the steering wheel that makes it easy to drive.

**What is a GPU and why do we use it?**
A GPU (Graphics Processing Unit) is a chip originally designed for video games. It turns out it's also extremely fast at the kind of maths deep learning needs (lots of matrix multiplications happening simultaneously). Training on a GPU can be 10–50x faster than on a regular CPU.

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```
This code finds the GPU and tells it to only use as much memory as it needs — instead of grabbing all of it at once. This prevents crashes when other programs also need the GPU.

**What is Mixed Precision?**
```python
mixed_precision.set_global_policy('mixed_float16')
```
Numbers in computers can be stored with different levels of detail. `float32` uses 32 bits (very precise), `float16` uses 16 bits (less precise but uses half the memory and is faster). Mixed precision uses float16 during most calculations but float32 for the final important steps. It's like doing rough work in pencil but writing the final answer in pen.

**What is a SEED and why do we set it?**
```python
SEED = 83
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
Machine learning involves randomness (random shuffling, random weight initialisation, etc.). If you set a seed, the "randomness" becomes *reproducible* — running it again gives the same results. Think of it like a dice that always rolls the same sequence of numbers if you start from the same point. This is important so experiments can be repeated and compared fairly.

**Key Hyperparameters explained:**

| Parameter | Value | What it means |
|---|---|---|
| `IMG_SIZE` | (300, 300) | All images are resized to 300x300 pixels |
| `BATCH_SIZE` | 64 | Train on 64 images at a time |
| `NUM_CLASSES` | 7 | 7 lizard species to classify |
| `VAL_SPLIT` | 0.15 | 15% of data kept for validation (testing during training) |
| `EPOCHS_FROZEN` | 50 | Max 50 rounds of training in Phase 1 |
| `EPOCHS_UNFROZ` | 35 | Max 35 rounds of training in Phase 2 |
| `LR_FROZEN` | 0.001 | Learning rate (how big each learning step is) in Phase 1 |
| `LR_UNFROZ` | 0.00001 | Much smaller learning rate for Phase 2 (careful fine-tuning) |

**What is a Learning Rate?**
Imagine you're walking blindfolded trying to find the lowest point in a valley. The learning rate decides how big each step you take is. Too big → you overshoot and miss the lowest point. Too small → it takes forever to get there. Finding the right balance is crucial.

---

### ❓ Possible Questions — Section 1

**Q: Why do you use EfficientNetV2-S specifically?**
A: EfficientNetV2-S was chosen because it's designed to be very efficient on smaller datasets. The "S" stands for Small — it's lighter and trains faster than larger models while still being very accurate. For a Kaggle competition with a limited number of lizard images, it performs better than larger networks that might overfit.

**Q: What is overfitting?**
A: Overfitting is when the model memorises the training data instead of learning general patterns. Like a student who memorises answers to past exams but can't answer questions they haven't seen before. The model performs great on training data but poorly on new, unseen data. We fight this with dropout layers, regularisation, data augmentation, and class weights.

**Q: Why set `BATCH_SIZE = 64` specifically?**
A: Larger batches give the model more information per update (more stable), but use more memory. 64 is a common sweet spot — big enough for stable training, small enough to fit in GPU memory. The comment in the code even says "More steps/epoch for better learning."

**Q: What is an epoch?**
A: One epoch = one complete pass through the entire training dataset. If you have 500 training images and a batch size of 64, one epoch = about 8 steps (500/64). After 50 epochs, the model has seen every training image 50 times.

---

## Section 2 — Data Loading & Quick EDA

### What's happening?
We load two CSV files (`train.csv` and `test.csv`) and check our dataset. EDA stands for **Exploratory Data Analysis** — basically, *getting to know your data before you start*.

```python
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
```

**What is a DataFrame?**
A DataFrame (from the Pandas library) is just a table — like an Excel spreadsheet in Python. Each row is one image, with columns for the image ID and its label (which lizard species it is).

**Building file paths:**
```python
def get_filepath(row):
    folder = CLASS_NAMES[int(row['label'])]
    return str(TRAIN_DIR / folder / row['id'])
```
The images are stored in folders named after each species. This function builds the full file path for each image by combining the folder (species name) + the image filename.

**Checking for missing files:**
```python
missing = train_df[~train_df['filepath'].apply(os.path.exists)]
```
This checks that every image file the CSV references actually exists on disk. The `~` is a NOT operator — it finds rows where the file does NOT exist.

**Class distribution:**
This prints how many images we have per lizard species. This is very important — if one species has 300 images and another has 30, the model will be biased toward the species with more data.

---

### ❓ Possible Questions — Section 2

**Q: What is EDA and why is it important?**
A: EDA (Exploratory Data Analysis) means understanding your data before building a model. You check for missing files, imbalanced classes, corrupted images, etc. It's important because garbage in = garbage out — if you don't understand your data, you'll build a bad model without knowing why.

**Q: What is class imbalance and why is it a problem?**
A: Class imbalance is when some categories have far more examples than others. In our dataset, there's an imbalance ratio printed (e.g., 3.5x means the biggest class has 3.5x more images than the smallest). If we don't handle this, the model learns to just predict the most common class and ignores the rare ones. It gets a high overall accuracy but a terrible F1 score. We fix this with **class weights** (Section 4).

---

## Section 3 — Train/Val Split & Data Pipeline

### What's happening?
We split our data into two groups and build an efficient data pipeline.

**The Split:**
```python
train_split, val_split = train_test_split(
    train_df, test_size=0.15, stratify=train_df['label'], random_state=SEED
)
```
85% of images go to **training** (the model learns from these). 15% go to **validation** (we test the model on these during training to see if it's improving).

**Why "stratified"?**
The `stratify=train_df['label']` argument ensures both the training and validation sets have the *same proportional mix of species*. Without this, you might accidentally put all images of one species into the training set and have none in validation — then you can't evaluate how well the model learned that species.

**The tf.data pipeline:**
```python
def parse_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label
```
This function:
1. Reads the raw file from disk
2. Decodes the JPEG (turns compressed bytes into pixel values)
3. Resizes to 300x300 (all images must be the same size for the neural network)

**Important: No rescaling to [0,1] here!**
Usually images are normalised to 0–1 range. But EfficientNet does its own internal preprocessing, so we deliberately keep the raw 0–255 range. Changing it would break the model's internal assumptions.

**`.shuffle()`, `.batch()`, `.prefetch()`:**
- **shuffle**: Randomly reorders images each epoch so the model doesn't memorise the order
- **batch**: Groups 64 images together for each training step
- **prefetch(AUTOTUNE)**: While the GPU trains on batch N, the CPU is already loading batch N+1. This keeps the GPU busy instead of waiting. Like a conveyor belt in a factory.

**One-hot encoding:**
```python
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))
```
Labels go from a single number (e.g., `3`) to a vector of 7 zeros with a 1 in the right position (e.g., `[0, 0, 0, 1, 0, 0, 0]`). This is required for the categorical crossentropy loss function we use.

---

### ❓ Possible Questions — Section 3

**Q: Why do we need a validation set if we already have a test set?**
A: The test set is used for the final Kaggle submission — we never look at its labels. The validation set is used during training to see how the model is performing on unseen data *so we can make decisions* (stop early, reduce learning rate, etc.). If we used the test set for this, we'd be "cheating" and the final score wouldn't be trustworthy.

**Q: Why do we shuffle training data but NOT validation data?**
A: Shuffling training data prevents the model from learning the order of images rather than the actual content. But validation data just needs to be evaluated — the order doesn't matter since we're just measuring performance, not learning.

**Q: What does AUTOTUNE do?**
A: `tf.data.AUTOTUNE` tells TensorFlow to automatically figure out the optimal number of parallel operations for loading data, based on your hardware. Instead of you guessing "use 4 CPU threads", TensorFlow tunes it dynamically at runtime.

**Q: Why resize all images to the same size?**
A: Neural networks have fixed input dimensions — like a specific-sized hole. Every image must fit exactly. The EfficientNetV2-S was designed for 300x300 input, so we resize all images to that exact size regardless of their original dimensions.

---

## Section 4 — Class Weights

### What's happening?
We calculate weights to compensate for class imbalance.

```python
cw_array = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train)
class_weight_dict = {i: cw_array[i] for i in range(NUM_CLASSES)}
```

**The analogy:**
Imagine you're a teacher grading tests, but 90% of your students are from one school and 10% from another. If you only care about overall pass rate, you naturally focus more on the big school. Class weights force you to care equally about both.

If species A has 200 images and species B has 50 images, species B gets a weight of 4x. This means every time the model makes a mistake on species B, the penalty is 4x larger — forcing the model to pay more attention to the rare classes.

The formula: `weight = total_samples / (num_classes × samples_in_class)`

---

### ❓ Possible Questions — Section 4

**Q: Could you just duplicate the rare class images instead of using class weights?**
A: Yes, that's called oversampling — it's another valid technique. But duplicating images means the model sees the exact same image multiple times, which can cause overfitting on those specific images. Class weights achieve the same balancing effect without introducing fake data. The teacher even might say "do you know why duplicating might be a bad idea?" — the answer is exactly that: *it's not real data anymore, and the model might overfit to specific examples rather than learning the general pattern.*

**Q: Why is F1 Score the evaluation metric and not just accuracy?**
A: Accuracy would be misleading with imbalanced classes. If 80% of test images are Green Iguanas, a model that always predicts "Green Iguana" gets 80% accuracy while being completely useless. F1 Score (Macro) calculates precision and recall for each class separately, then averages them — giving equal importance to every class regardless of how many examples it has.

**Q: What is Precision and Recall?**
A: 
- **Precision**: Of all the times the model said "this is a Green Anole", how many times was it actually right? (Avoid false alarms)
- **Recall**: Of all the actual Green Anoles in the dataset, how many did the model find? (Avoid missing things)
- **F1**: The harmonic mean of both. High F1 means you're both precise AND catching most cases.

---

## Section 5 — Model Architecture

### What's happening?
This is the most important section — we define the brain of our system.

**The Big Idea — Transfer Learning:**
EfficientNetV2-S was trained by Google on **ImageNet** — 14 million images across 1000 categories. It already knows how to recognise edges, textures, shapes, patterns, fur, scales, eyes — basically the building blocks of visual recognition.

We take this pre-trained model, remove its "head" (the final layer that classifies into 1000 categories), and replace it with our own head that classifies into 7 lizard species.

```python
base = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False  # Freeze the backbone — don't change its weights
```

**The Architecture Layer by Layer:**

```
Input (300×300×3 RGB image)
    ↓
RandomFlip (horizontal only)
    ↓
RandomRotation (±29°)
    ↓
RandomZoom (±10%)
    ↓
RandomContrast (±15%)
    ↓
RandomBrightness (±15%)
    ↓
EfficientNetV2-S backbone (frozen — just extracting features)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (40%)
    ↓
Dense (256 neurons, ReLU activation)
    ↓
Dropout (30%)
    ↓
Dense (7 neurons, Softmax activation) ← Final prediction
```

**Data Augmentation Layers (the Random* layers):**
These artificially create variations of the training images to simulate real-world variation:
- **RandomFlip('horizontal')**: Mirrors the image left-right. A lizard facing left is still the same species facing right. We do NOT flip vertically because an upside-down lizard is an unnatural/destructive transformation.
- **RandomRotation(0.08)**: Rotates slightly (~29°). Lizards are photographed at various angles.
- **RandomZoom(0.1)**: Zooms in or out by up to 10%.
- **RandomContrast(0.15)**: Adjusts contrast by ±15%.
- **RandomBrightness(0.15, value_range=(0, 255))**: Adjusts brightness. The `value_range=(0, 255)` is critical — it tells Keras our images are in 0–255 range, not 0–1. Getting this wrong would produce extreme black or white images.

**Why is augmentation in the model, not in the data pipeline?**
Keras augmentation layers are automatically disabled during validation and testing — they only activate during training. If augmentation was in the data pipeline (tf.data), you'd need to manually handle this. Also, GPU augmentation is faster than CPU augmentation.

**GlobalAveragePooling2D:**
The EfficientNet backbone outputs a 3D feature map (a grid of activations). GlobalAveragePooling2D collapses this grid by averaging each feature across all positions — turning a 10×10×1280 tensor into a 1280-dimensional vector. This is like summarising a book by averaging the key ideas from every paragraph.

**Dropout:**
```python
x = layers.Dropout(0.4)(x)  # 40% dropout
x = layers.Dropout(0.3)(x)  # 30% dropout
```
During training, Dropout randomly turns off 40% (or 30%) of neurons on each pass. This forces the network not to rely on any single neuron — it must learn redundant, robust representations. During testing, all neurons are active. This significantly reduces overfitting.

**Dense layers:**
Fully connected layers. Every neuron connects to every neuron in the previous layer. The first Dense (256 neurons) learns high-level combinations of features. The second Dense (7 neurons) makes the final prediction for each class.

**ReLU vs Softmax:**
- **ReLU**: `max(0, x)` — simple activation that introduces non-linearity, allows the model to learn complex patterns
- **Softmax**: Converts raw scores into probabilities that sum to 1. If the model outputs [0.02, 0.01, 0.03, 0.85, 0.02, 0.04, 0.03], the highest number (0.85 = Desert Iguana) is the prediction.

**Label Smoothing (0.1):**
Normally, the target for a "Green Anole" image is [0, 0, 0, 0, 1, 0, 0] — perfectly confident. Label smoothing softens this to [0.014, 0.014, 0.014, 0.014, 0.914, 0.014, 0.014]. It tells the model: "Be mostly sure, but not 100% certain." This improves generalisation because the model learns to be calibrated rather than overconfident.

**L2 Regularisation:**
```python
kernel_regularizer=keras.regularizers.l2(1e-4)
```
Adds a small penalty to the loss for large weights. This discourages the model from making any single weight extremely large (which would mean the model relies too heavily on one feature). Another tool against overfitting.

**dtype='float32' on the output layer:**
```python
outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
```
Because we're using mixed precision (float16 for most layers), the final prediction layer is explicitly kept at float32 for numerical stability. Softmax probabilities at float16 precision can become inaccurate.

---

### ❓ Possible Questions — Section 5

**Q: What is Transfer Learning and why is it useful here?**
A: Transfer learning reuses knowledge a model learned from one task to help with another. EfficientNetV2-S learned to recognise visual patterns from 14 million images. Rather than training from scratch (which would need millions of lizard photos and weeks of compute), we "transfer" that general visual knowledge and just teach it the specific differences between 7 lizard species. It's like hiring a doctor who studied medicine for 10 years to specialise in one disease — they don't need to relearn biology from scratch.

**Q: Why did you remove vertical flips?**
A: Vertical flips would create upside-down lizard images. Lizards in photos are always right-side up — an upside-down lizard image is not a real-world scenario. Training on upside-down images would introduce misleading patterns that the model would never encounter during real predictions. This is called **destructive augmentation** — it makes the data less realistic, not more.

**Q: What is the difference between the backbone and the head?**
A: The **backbone** (EfficientNetV2-S) is the feature extractor — it reads the image and converts it into a rich set of features (like "there are scales here", "the colour is green", "there's a crest shape"). The **head** is the classifier — it takes those features and decides which lizard species they belong to. The backbone does the heavy lifting; the head makes the final call.

**Q: Why is `base(x, training=False)` in Phase 1?**
A: Setting `training=False` tells the backbone to behave as if it's always in "evaluation mode" — specifically, BatchNorm layers use their pre-learned statistics instead of updating them. This is important because during Phase 1, the backbone is frozen and shouldn't be learning or changing anything. If we said `training=True`, BatchNorm would try to update its statistics based on our small dataset, which would corrupt the pre-trained knowledge.

---

## Section 6 — Training Callbacks

### What's happening?
Callbacks are "automatic assistants" that monitor training and take actions when certain things happen.

**EarlyStopping:**
```python
EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
```
Stops training early if the validation metric hasn't improved in 8 epochs. Analogy: if a student's exam scores haven't improved in 8 practice tests, stop studying that way — you've peaked. `restore_best_weights=True` means it goes back to the best version, not the last version.

**ReduceLROnPlateau:**
```python
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8)
```
If `val_loss` doesn't improve for 3 epochs, it halves the learning rate (`factor=0.5`). Like taking smaller steps when you keep overshooting the target. `min_lr=1e-8` ensures the learning rate never becomes so tiny that training completely stops.

**ModelCheckpoint:**
```python
ModelCheckpoint(filepath=str(MODEL_SAVE), monitor='val_accuracy', save_best_only=True)
```
Saves the model to disk whenever it achieves a new best validation score. This means even if the model gets worse after more training, you still have the best version saved. Like taking a photo whenever you reach a new personal best in a race.

**Why Phase 1 monitors `val_accuracy` but Phase 2 monitors `val_loss`:**
- `val_accuracy` is a better signal when the model is still learning quickly (Phase 1 — head training).
- `val_loss` is a smoother signal because it's a continuous number (not just right/wrong), making it better for fine-tuning small adjustments (Phase 2).

---

### ❓ Possible Questions — Section 6

**Q: What's the difference between val_loss and val_accuracy?**
A: `val_accuracy` is the percentage of correct predictions (0–100%). `val_loss` is a continuous number measuring how "wrong" the model was — it accounts for how *confident* the wrong predictions were. A model that's barely wrong has a lower loss than one that's very confidently wrong. This makes val_loss a more sensitive and smoother signal for fine-tuning.

**Q: Why not just always use val_loss for monitoring?**
A: In Phase 1, the model is learning rapidly and val_accuracy gives a clear, interpretable signal of progress. Also, EarlyStopping on val_accuracy in Phase 1 naturally waits for accuracy plateaus rather than tiny loss improvements. It's a design choice about what signal is most meaningful at each phase.

---

## Section 7 — Phase 1: Train Head (Frozen Backbone)

### What's happening?
The backbone is frozen (its weights don't change). We only train the new layers we added (the head).

**The Analogy:**
Think of the backbone as a professional photographer who already knows how to use cameras, lighting, and composition. In Phase 1, we're just teaching them *which lizard characteristics to look for* — but they already have all their photography skills intact. The backbone stays exactly as Google trained it.

```python
history1 = model.fit(
    train_ds,
    epochs=EPOCHS_FROZEN,     # Up to 50 epochs
    validation_data=val_ds,
    class_weight=class_weight_dict,  # Apply class balancing
    callbacks=make_callbacks_phase1(),
    verbose=1
)
```

Training stops early (before 50 epochs) if EarlyStopping triggers. The model saves its best weights automatically.

**Expected result:** ~75% validation accuracy. The model learns basic lizard classification with just the new head layers being trained.

---

### ❓ Possible Questions — Section 7

**Q: Why train with a frozen backbone first instead of training everything at once?**
A: The new head layers are initialised with random weights. If you immediately allow the backbone to change too, the random head layers could "corrupt" the carefully pre-trained backbone weights through a process called catastrophic forgetting. By freezing the backbone first, the head learns to work with the backbone's existing features. Then in Phase 2, we can safely fine-tune them together.

**Q: What is catastrophic forgetting?**
A: When a neural network learns a new task, it can "forget" what it learned on the old task if the new training signal is very different. By freezing the backbone in Phase 1, we prevent the lizard data (small dataset) from overwriting the ImageNet knowledge (huge dataset) before the head is ready.

---

## Section 8 — Phase 2: Fine-tune Top 40 Layers

### What's happening?
Now we "unfreeze" the top part of the backbone and let it also adapt to lizard images.

**The Key Insight — Why only the top 40 layers?**
Neural networks learn features hierarchically:
- **Early layers**: Basic features — edges, corners, colours, simple textures (universal, same for all images)
- **Middle layers**: More complex patterns — scales, skin texture, shapes
- **Top layers (last 40)**: High-level, task-specific features — the differences between species

We only fine-tune the top (deepest) layers because those are the most task-specific. The early layers already have perfect universal features and we don't want to mess with them.

**Critical Fix — Freezing BatchNorm layers:**
```python
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
```
**This is the most important fix in v4.** BatchNorm layers store statistics (mean and variance) calculated from ImageNet's millions of images. If we allow them to update on our small lizard dataset, those statistics will be corrupted. With corrupted BatchNorm, Phase 2 actually made the model *worse* in earlier versions. By keeping BatchNorm frozen, we preserve those statistics and fine-tuning works correctly.

**Lower Learning Rate:**
`LR_UNFROZ = 1e-5` (0.00001) — 100x smaller than Phase 1's 0.001. We're making tiny, careful adjustments to a network that's already good, not relearning from scratch.

**Expected result:** Validation accuracy should *improve* from Phase 1's ~75% to 78–80%+. In previous broken versions (without BatchNorm freezing), Phase 2 made the model worse — this is the core bug that v4 fixes.

---

### ❓ Possible Questions — Section 8

**Q: What is BatchNormalization and why do we freeze it?**
A: BatchNorm is a layer that normalises the values flowing through the network — it keeps them in a stable range so training is more stable. It stores a "running average" of what values it has seen (mean and variance). These statistics were calculated from ImageNet's 14 million images. If we update them with our small lizard dataset, the statistics become wrong for the backbone's general knowledge. Keeping them frozen preserves ImageNet's statistics, which keeps the backbone stable.

**Q: Why is the learning rate so much lower in Phase 2?**
A: The backbone has carefully tuned weights from Google's training. Large learning rate steps could drastically change (or destroy) these weights. Using a very small learning rate (1e-5) makes tiny, surgical adjustments that gently adapt the backbone to lizard features without destroying what it already knows.

**Q: What does `base_model.layers[:-40]` mean?**
A: In Python, `[:-40]` means "everything except the last 40". So `base_model.layers[:-40]` selects all layers from the beginning up to but not including the final 40 layers. We freeze these (keep them unchanged) and only allow the last 40 layers to be trained.

---

## Section 9 — Training History Visualization

### What's happening?
We plot the accuracy and loss curves across both training phases to visually inspect the training process.

```python
def plot_history(h1, h2):
    # Plots Phase 1 and Phase 2 accuracy/loss on the same graphs
```

The graphs show:
- **Training accuracy/loss**: How well the model does on training data
- **Validation accuracy/loss**: How well the model does on unseen validation data

**What good training looks like:**
- Both training and validation metrics improve together
- Validation doesn't diverge far below training (that's overfitting)
- Curves flatten out — the model has converged

**What to look for between phases:**
- Phase 1 ends where Phase 2 begins (dotted lines)
- Phase 2 should continue improving from where Phase 1 left off
- If Phase 2 shows the val_accuracy *going up* (not down), the BatchNorm fix worked

---

### ❓ Possible Questions — Section 9

**Q: What does it mean if validation loss goes up while training loss goes down?**
A: That's the classic sign of overfitting. The model is getting better at memorising the training set but generalising worse to new data. We fight this with dropout, regularisation, augmentation, early stopping, and class weights.

**Q: Why save the plots to files?**
A: So you can share them in reports or presentations without re-running the (very slow) training process. Training takes hours — you only want to do it once.

---

## Section 10 — Evaluation & Confusion Matrix

### What's happening?
We evaluate the trained model on the validation set and visualise its mistakes.

**Getting predictions:**
```python
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
```
`np.argmax` finds the index of the highest probability. If Softmax output is `[0.02, 0.01, 0.03, 0.85, 0.02, 0.04, 0.03]`, argmax returns `3` (Desert Iguana).

**Classification Report:**
For each lizard species, it shows:
- **Precision**: Of all images the model said were this species, what % were actually this species?
- **Recall**: Of all images that ARE this species, what % did the model correctly identify?
- **F1-score**: Harmonic mean of precision and recall
- **Support**: How many images of this species were in the validation set

**The Confusion Matrix:**
This is a grid where:
- Rows = the TRUE species (what the lizard actually is)
- Columns = PREDICTED species (what the model said)
- Diagonal = correct predictions (we want these to be high)
- Off-diagonal = mistakes (what it confused with what)

**Example:** If row "Green Anole" has a high number in column "Brown Anole", the model often confuses Green Anoles with Brown Anoles — which makes sense since both are small anole lizards.

```
                 Predicted →
                 BlkSp  BrnA  CuKn  Dsrt  GrnA  GrnI  LsAn
Actual ↓ BlkSp [  45     0     0     0     0     2     0  ]   ← 45 correct, 2 wrong
         BrnA  [   0    38     2     0     1     0     0  ]
         CuKn  [   0     1    42     0     0     0     0  ]
         ...
```

The diagonal numbers should be as high as possible. Off-diagonal numbers show where the model gets confused.

---

### ❓ Possible Questions — Section 10

**Q: What is a Confusion Matrix?**
A: A confusion matrix is a table that shows where the model gets confused. Each row represents the actual class, each column represents the predicted class. The diagonal (top-left to bottom-right) shows correct predictions. Everything off the diagonal is a mistake — and it tells you specifically which classes are being mixed up with which.

**Q: What's the difference between Macro and Weighted F1?**
A: 
- **Macro F1**: Calculates F1 for each class separately, then averages them equally. Every class counts the same regardless of size. This is our Kaggle metric.
- **Weighted F1**: Averages F1 scores weighted by how many examples each class has. Bigger classes influence the score more.
- We use **Macro** because we care equally about correctly classifying all 7 species, even the rare ones.

**Q: Why do we run evaluation on the validation set and not the test set?**
A: The test set has no labels — those are what Kaggle grades us on. We only use the test set once, for final submission. The validation set is our internal benchmark.

---

## Section 11 — Test Predictions with Test-Time Augmentation (TTA)

### What's happening?
We make predictions on the test set (the images with no labels) for Kaggle submission, using **Test-Time Augmentation (TTA)**.

**What is TTA?**
Normally you show the model one image and take one prediction. With TTA, you show the same image multiple times with slight variations (flipped, rotated, etc.) and average all the predictions together.

**The Analogy:**
Imagine you're trying to identify a lizard species but you're not sure. You look at it from the front, then the side, then slightly rotated. You combine all these perspectives to make a more confident final decision. TTA does the same thing — averaging predictions from multiple "views" of the same image reduces the impact of individual noise.

```python
test_preds = predict_tta(model, test_images, n_aug=3)  # Original + 3 augmented versions
```

**Why 1–2% boost?**
Machine learning models can be sensitive to exact image positioning. By averaging over augmented versions, you smooth out this sensitivity and get a more stable prediction, especially for images that are near the decision boundary between two classes.

**The helper functions `load_test_image` and `predict_tta`:**
These functions need to be defined before this cell runs — `load_test_image` loads a test image (no label, just the image) and `predict_tta` runs the model multiple times with augmented versions of each image and averages the results.

---

### ❓ Possible Questions — Section 11

**Q: Why do we need TTA if we already have data augmentation during training?**
A: Training augmentation makes the model more robust by exposing it to variations. TTA is different — it's used at prediction time to make individual predictions more robust. Think of training augmentation as practice and TTA as double-checking your answer before submitting.

**Q: What is a Kaggle submission?**
A: Kaggle is a data science competition platform. We submit a CSV file with our predicted labels for the test images. Kaggle compares our predictions to the hidden true labels and gives us a score. Our metric is F1 Macro.

---

## Section 12 — Model Analysis & Expected Improvements

This section summarises the 5 key fixes that v4 introduced over earlier versions:

| Fix | What it does | Expected impact |
|---|---|---|
| **Fix A** | Freeze BatchNorm during fine-tuning | +3–5% (Phase 2 now improves instead of degrading) |
| **Fix B** | Consistent val_loss monitoring in Phase 2 | Smoother, more reliable training |
| **Fix C** | Keras augmentation layers with correct brightness range | Correct augmentation (was broken before) |
| **Fix D** | Test-Time Augmentation (TTA) | +1–2% Kaggle score |
| **Fix E** | EfficientNetV2-S (vs EfficientNetB3) | +2–3% better small dataset performance |

**Expected Kaggle score: 80–83% F1 Macro**

---

## Section 13 — GenAI Reflection

This section documents that Claude (AI) was used for debugging and architecture advice. The key decisions (which hyperparameters, which architecture choices, result analysis) were made by the team.

---

## 🔑 Master Cheat Sheet — Most Likely Teacher Questions

| Question | Answer |
|---|---|
| What is Transfer Learning? | Reusing a model trained on a big dataset (ImageNet) to solve a different but related problem (lizard classification) |
| What is overfitting? | Model memorises training data but fails on new data. Solved with dropout, regularisation, augmentation, early stopping |
| Why freeze BatchNorm? | BatchNorm stores statistics from ImageNet training. Updating them on a small lizard dataset corrupts the backbone |
| What does a Confusion Matrix show? | Rows = true class, Columns = predicted class. Diagonal = correct. Off-diagonal = mistakes |
| Why F1 and not accuracy? | Imbalanced classes make accuracy misleading. F1 gives equal weight to all classes |
| What is Dropout? | Randomly disables neurons during training to prevent overfitting |
| What is one epoch? | One full pass through all training images |
| Why no vertical flips? | Upside-down lizards don't appear in nature — this augmentation is destructive and misleading |
| Why stratified split? | Ensures both training and validation sets have the same proportion of each species |
| What is label smoothing? | Softens target labels from hard 0/1 to soft values, making the model less overconfident |
| What is a learning rate? | How big each training step is. Too high = overshooting. Too low = too slow |
| Why two training phases? | Phase 1 safely trains the new head. Phase 2 fine-tunes the top of the backbone with a small LR |
| What is GlobalAveragePooling? | Collapses the 3D feature map from the backbone into a 1D vector by averaging each feature across all positions |
| What is Softmax? | Converts raw scores to probabilities that sum to 1. The highest probability = the predicted class |
| What is prefetch? | Pre-loads the next batch of data while the GPU is still training on the current one, keeping the pipeline efficient |
| Why class weights? | Some species have fewer images. Class weights penalise mistakes on rare classes more heavily, forcing the model to learn them |
| Why mixed precision? | float16 uses half the memory and is faster on modern GPUs, with minimal accuracy loss |
| What is EfficientNetV2-S? | A state-of-the-art image classification network by Google, pre-trained on ImageNet. "S" = Small, good for small datasets |

---

*Good luck with your defence! You've got this. 🦎*
