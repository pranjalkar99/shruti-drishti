![Synapse ai](https://github.com/pranjalkar99/shruti-drishti/assets/106006087/e6811f64-ed88-48d0-a04d-d41bf85b4668)

[Main presentation](https://www.canva.com/design/DAGCJAS6jlU/LLS2hNUTG0Z_tMLmPdM1UQ/view?utm_content=DAGCJAS6jlU&utm_campaign=designshare&utm_medium=link&utm_source=editor)
# Shruti-Drishti: Bridging the Communication Gap for the Deaf Community in India 🌉🇮🇳

## Introduction 🙌

Shruti-Drishti is an innovative project aimed at addressing the communication gap between the deaf and non-deaf communities in South Asia, particularly in India. By leveraging deep learning models and state-of-the-art techniques, we strive to facilitate seamless communication and promote inclusivity for individuals with hearing impairments. 🌟

## DEMO VIDEO
![Demo for ISL based Sign Language Detection](https://img.youtube.com/vi/WsnubMzE8yM/0.jpg)


## Key Features ✨

1. **Sign Language to Text Conversion** 🖐️➡️📝: Our custom Transformer-based Multi-Headed Attention Encoder, powered by Google's Tensorflow Mediapipe, accurately converts sign language videos into text, overcoming challenges related to dynamic sign similarity.

2. **Text to Sign Language Generation** 📝➡️🖐️: Utilizing an Agentic LLM framework, Shruti-Drishti converts textual information into masked keypoints based sign language videos, tailored specifically for Indian Sign Language.


![Text2sign](https://github.com/pranjalkar99/shruti-drishti/assets/106006087/b76f1a8e-ca18-43b7-836c-3f6b3aa2e912)

3. **Multilingual Support** 🌐: Our app uses IndicTrans2 for multilingual support for all 22 scheduled Indian Languages. Accessibility is our top priority, and we make sure that everyone is included. 

4. **Content Accessibility** 📰🎥: Shruti-Drishti enables news channels and content creators to expand their user base by making their content accessible and inclusive through embedded sign language video layouts.

## Dataset Details 📊
Link to the Dataset: [INCLUDE Dataset](https://zenodo.org/records/4010759)

The INCLUDE dataset, sourced from AI4Bharat, forms the foundation of our project. It consists of 4,292 videos, with 3,475 videos used for training and 817 videos for testing. Each video captures a single Indian Sign Language (ISL) sign performed by deaf students from St. Louis School for the Deaf, Adyar, Chennai.

## Model Architecture 🧠

Shruti-Drishti employs two distinct models for real-time Sign Language Detection:

1. **LSTM-based Model** 📈: Leveraging keypoints extracted from Mediapipe for poses, this model utilizes a recurrent neural network (RNN) and Long-Short Term Memory Cells for evaluation.
   - Time distributed layers: Extract features from each frame based on the Mediapipe keypoints. These features capture spatial relationships between joints or movement patterns.
   - Sequential Layers: Allows the model to exploit the temporal nature of the pose data, leading to more accurate pose estimation across a video sequence.

2. **Transformer-based Model** 🔄: Trained through extensive experimentation and hyperparameter tuning, this model offers enhanced performance and adaptability. 
   - **Training Strategies:**
     1. Warmup: Gradually increases the learning rate from a very low value to the main training rate, helping the model converge on a good starting point in the parameter space before fine-tuning with higher learning rates.
     2. AdamW: An advanced optimizer algorithm that addresses some shortcomings of the traditional Adam optimizer and often leads to faster convergence and improved performance.
     3. ReduceLRonPlateau: Monitors a specific metric during training and reduces the learning rate if the metric stops improving for a certain number of epochs, preventing overfitting and allowing the model to refine its parameters.
     4. Finetuned VideoMAE: Utilizes the pre-trained weights from VideoMAE as a strong starting point and allows the model to specialize in recognizing human poses within videos.

We have also implemented the VideoMAE model, proposed in the paper ["VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training."](https://arxiv.org/abs/2203.12602) Fine-tuning techniques such as qLORA, peft, head and backbone fine-tuning, and only head fine-tuning were explored, with the latter proving to be the most successful approach.

## Solution Approach 🎯

Shruti-Drishti tackles the communication gap through a two-fold approach:

1. **Sign Language to Text**: Implementing a custom Transformer-based Multi-Headed Attention Encoder using Google's Tensorflow Mediapipe, we convert sign language videos into text while addressing challenges related to dynamic sign similarity.

2. **Text to Sign Language**: Utilizing an Agentic LLM framework, Shruti-Drishti converts textual information into masked keypoints based sign language videos, tailored specifically for Indian Sign Language.

## Action Plans 📋

1. **Pose-to-Text Implementation**: Develop and implement a Pose-to-Text model based on the referenced paper for the Indian Sign Language dataset, using Agentic langchain based state flow as the decoder stage for text-to-gloss conversion and merging masked keypoint videos.

2. **Custom Transformer Model Evaluation**: Assess the effectiveness of our custom Transformer/LSTM model on the Sign Language Dataset, focusing on accuracy and adaptability to dynamic signs.

3. **Multilingual App Development**: Create a user-friendly multilingual app serving as an interface for our Sign Language Translation services, ensuring easy interaction and adoption by both deaf and non-deaf users.

## Progress So Far ✅

- [x] Basic Deep Learning-based LSTM model for sign language recognition (Done)
- [x] Custom multi-headed attention-based encoder for sign language recognition for dynamic signs (Done)
- [x] Testing on the whole Indian dataset for our attention model (Done)
- [x] Implementing the pose-to-text using agentic framework (Langgraph) (Done)
- [x] Build multilingual app (Done)
- [ ] Build Demo and update repo (In Progress)

## Results 📈

### Transformers
![Results Image](https://github.com/pranjalkar99/shruti-drishti/assets/74347116/3541813b-52c2-4c10-a7ac-88096aac62b4)

For detailed results and insights, please refer to our [presentation slides](https://www.canva.com/design/DAF_IfblIbM/Hm_cvyUw6vNEf8-RXg68fg/edit?utm_content=DAF_IfblIbM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

### LSTM
(TODO)

## Other Links 🔗

- [Demo Video](https://www.youtube.com/watch?v=hR-aP7o53iQ)
- [Presentation Slides](https://www.canva.com/design/DAF_IfblIbM/Hm_cvyUw6vNEf8-RXg68fg/edit?utm_content=DAF_IfblIbM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- [Hasgeek Presentation Slides](https://www.canva.com/design/DAGABnVhHqw/d2T8fLDof94PabPlWoKHEg/edit?utm_content=DAGABnVhHqw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- [ISL History and its Need](https://islrtc.nic.in/history-0#:~:text=Indian%20Sign%20Language%20(ISL)%20is,material%20that%20incorporates%20sign%20language.)
- [Project Roadmap Github View](https://github.com/users/pranjalkar99/projects/2/views/2)
- [ISL Dataset](https://zenodo.org/records/4010759)
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base)
- [AI4Bharat Models](https://huggingface.co/ai4bharat)
- [Langgraph framework](https://python.langchain.com/docs/langgraph/)
- [Langchain framework](https://python.langchain.com/docs/get_started/introduction)

## Project Contributors 👥

- [Pranjal Kar](https://github.com/pranjalkar99/)
- [Bhaswata Choudhury](https://github.com/bhaswata08)
- [Samunder Singh](https://github.com/samthakur587)
