# Video Demo
<iframe width="1519" height="574" src="https://www.youtube.com/embed/WsnubMzE8yM" title="Demo for ISL based Sign Language Detection" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

# Problem Statement

## Addressing the Communication Gap between Deaf and Non-Deaf Communities in South Asia, Particularly India

**Overview:**

South Asia, including India, is home to a significant portion of the global deaf population. Despite estimates suggesting a high number of individuals affected by hearing loss, accessibility and sign language resources are limited in India. The National Association of the Deaf puts the number at 18 million, while other sources indicate it could be as high as 63 million.

The lack of accessibility and communication tools poses challenges for education and employment opportunities for the deaf community. Efforts to establish Indian Sign Language as an official language are ongoing, and the need for innovative solutions to bridge the communication gap persists.

# Solution

## Leveraging Deep Learning for Sign Language Translation

**Approach:**

Our proposed solution involves using deep learning models to facilitate communication:

1. **Sign Language to Text:**
   - Implement a custom Transformer-based Multi-Headed Attention Encoder using Google's Tensorflow Mediapipe for converting sign language videos into text.
   - Address challenges related to dynamic signs similarity by utilizing keypoint data generated by Google's Tensorflow Mediapipe.

2. **Text to Sign Language:**
   - Reproduce the methodology outlined in the paper [link to paper], specifically tailored for Indian Sign Language.
   - Utilize a Generative Adversarial Network (GAN) model to convert textual information into structural keypoints and further generate sign language videos.

**Use Cases:**

1. **Workplace and Educational Inclusion:**
   - Deploy the Sign Language Generation system in offices and educational institutions to facilitate seamless communication with the deaf and mute community.
   - Empower individuals with hearing impairments by providing them with equal opportunities for education and employment.

2. **Content Accessibility:**
   - Enable news channels and content creators to expand their user base by making their content accessible and inclusive.
   - Offer services to embed sign language video layouts for content, fostering a more inclusive society and promoting equal participation.

# Action Plans

1. **Pose-to-Text Implementation:**
   - Develop and implement a Pose-to-Text model based on the paper [link to paper] for the Indian Sign Language dataset.
   - Utilize Gemini/GPT-4 as the decoder stage for text-to-gloss conversion.

2. **Custom Transformer Model Evaluation:**
   - Assess the effectiveness of our custom Transformer model on the Sign Language Dataset, focusing on accuracy and adaptability to dynamic signs.

3. **Multilingual App Development:**
   - Create a user-friendly multilingual app serving as an interface for our Sign Language Translation services.
   - Ensure the app facilitates easy interaction and adoption by both deaf and non-deaf users.

# Progress So Far

- [x] Basic Deep Learning-based LSTM model for sign language recognition (Done)
- [x] Custom multi-headed attention-based encoder for sign language recognition for dynamic signs (Done)
- [ ] Testing on the whole Indian dataset for our attention model (In Progress)
- [ ] Implementing the pose-to-text and the implementation of the paper (In Progress)
- [ ] Build multilingual app (To Do)

## Proposed Workflow (credits to the Open Source Paper)
![image](https://github.com/pranjalkar99/shruti-drishti/assets/74347116/4636a003-09f4-4953-92ad-c3df4b9fea1e)



#### Project Contributors:
**Pranjal Kar** (https://github.com/pranjalkar99/)
** Ashish Sarmah** (https://github.com/NoisyBotDude)
**Rajdeep Nath** (https://github.com/RAZDYP)
**Bhaswata Choudhury** (https://github.com/bhaswata08)
**Samunder Singh** (https://github.com/samthakur587)



(we thank the authors of the paper https://aclanthology.org/2023.at4ssl-1.3.pdf for the architectural flow, and workflow, our open source project is aimed at research for Indian usecases)
**Note:**
The development and research is under progress.
