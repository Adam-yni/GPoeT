# Tiny Poetic LLM 🎭✨

## About the Project

A small personal project to create a **tiny language model (LLM)** for generating poetry in French. The model contains **40M parameters** and was entirely trained on a **Kaggle notebook** (free tier with an NVIDIA P100 GPU, 16GB VRAM). 

### Features 🌟
- ✅ **Accurate grammar**: Handles gender and plural correctly.  
- 📝 **Coherent verses**: The output is mostly consistent and meaningful (though very abstract at times..).  
- 🎛️ **Adjustable creativity**: A temperature setting below `0.4` can yield surprisingly interesting results!  
- ❌ **No rhymes (yet)**: The current model rarely rhymes. I hope that a second training phase using reinforcement learning will address this.  

---

## Dataset 📚

The model was trained on a custom dataset containing **2.2M characters** compiled from public domain poetry collections on **Project Gutenberg**, featuring:  
- **Victor Hugo**  
- **Charles Baudelaire**  
- **Paul Verlaine**

### Tokenizer ✂️  
The tokenizer was built using **Byte Pair Encoding (BPE)** with a **tiny vocabulary of 128 tokens**, including:  
- Uppercase and lowercase letters.  
- Frequently recurring character groups.  

### Positional Encoding 🌀  
Used **RoPE (Rotary Positional Encoding)** for efficient handling of relative token positions.  

---

## Training 🏋️‍♂️

- Trained entirely on **Kaggle's free GPU environment**.  
- The model is compact and focuses on generating stylistically accurate poetry.  

---

## Next Steps 🚀  

### Reinforcement Learning 🤖  
Developing a reinforcement learning algorithm inspired by **GFlowNets** to enhance rhyme generation and improve thematic consistency.  

---

## Why Poetry? 🎨  

Poetry is a challenging and nuanced art form. This project explores how a simple LLM can generate creative text. Like many of us reading a poem in school and struggling to decipher its meaning, this tiny LLM might be creating verses that hold an abstract beauty we can't always comprehend.  

Who knows? Maybe it's writing poetry that resonates beyond its apparent limits!  

---

## Example Outputs ✍️  

With a **temperature < 0.4**, the model sometimes produces surprising and intriguing verses. Here's a sample output (with temperature = 0.1):  

> *"Avec la mort des cheveux d'or,
        Le soir étend la mort.

On voit dans l'ombre une âme invisible à l'homme.
L'homme est un soleil d'une fleur qui se déchire,
        L'amour est toujours nue 
L'âme dans l'ombre des cieux s'en va fait la nuit.
L'homme est un vent se dresse et se lamente.
        L'homme est seul et la mort.
L'homme est un coeur en chemin s'élance et s'envole.
L'homme est un vent se dresse à l'ombre des cieux.
        L'amour est un ciel bleu 
L'ombre des fleurs s'envole, et l'on voit s'étend"*  
With the death of the golden hairs,
        Evening spreads death.

We see in the shadows a soul invisible to man.
Man is a sun from a flower that is torn apart,
        Love is always naked 
The soul in the shadow of the heavens goes away at night.
Man is a wind that rises and laments.
        Man is alone and death.
Man is a heart on the way soars and flies away.
Man is a wind standing in the shadow of the heavens.
        Love is a blue sky 
The shadow of the flowers flies away, and we see it extend
---

### Contributions & Feedback 💬  
This project is a work in progress—feel free to suggest ideas or improvements!  
