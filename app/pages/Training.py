import streamlit as st

st.set_page_config(page_title="Training and Data", layout="wide")

st.title("Training and Data")

st.markdown("""
We apply a **curriculum learning approach**, starting with simple and clear examples to help the model learn 
fundamental visual patterns before moving to more complex, real-world cases.
""")

#____________________________Section 1________________________________________________________________________
st.markdown(
    "<h4 style='margin-top:1px;'>1. Dataset and Training V1</h4>", 
    unsafe_allow_html=True
)
st.markdown(
    "View Dataset: [v1_malaysian_car_plate-fkgj1](https://app.roboflow.com/yolo-zmazg/v1_malaysian_car_plate-fkgj1/1)"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.image("images/dataset_demo/1.jpg", caption="Close-up sample 1", use_container_width=True)
with col2:
    st.image("images/dataset_demo/2.jpg", caption="Close-up sample 3", use_container_width=True)
with col3:
    st.image("images/dataset_demo/3.jpg", caption="Close-up sample 2", use_container_width=True)

st.markdown("""
Using YOLO weights pretrained on the **COCO dataset**, we initiated our curriculum with a **simple, clean dataset** 
containing **close-up and clear shots of Malaysian car plates**. The goal of this initial phase was to help the model learn the **basic geometry, proportions, and visual characteristics** 
of Malaysian license plates before introducing more challenging variations such as **distance, glare, and occlusion**.

After this stage of training, the model demonstrated early signs of **generalizing plate-like patterns**, 
occasionally identifying rectangular regions with text (such as signboards) as license plates — 
a typical behavior observed in the early stages of curriculum learning.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Precision", value="0.986")
with col2:
    st.metric(label="Recall", value="0.944")
with col3:
    st.metric(label="mAP@50", value="0.948")
with col4:
    st.metric(label="mAP@50-95", value="0.646")

st.markdown("""
**Dataset and Training Summary**

• Dataset Size: 72 images  
• Train: 41 | Validation: 18 | Test: 13  
• Training: 50 epochs  
• Image Size: 1280  
• Elapsed Time: 0:05:36 (336s)
""")
st.divider()
#____________________________Section 1________________________________________________________________________


#____________________________Section 2________________________________________________________________________
st.markdown(
    "<h4 style='margin-top:1px;'>2. Dataset and Training V2</h4>", 
    unsafe_allow_html=True
)
st.markdown(
    "View Dataset: [v2_leegame-kl_drive_morning-ew86j](https://app.roboflow.com/yolo-zmazg/v2_leegame-kl_drive_morning-ew86j/models)"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.image("images/dataset_demo/5.jpg", caption="Clear on the road images", use_container_width=True)
with col2:
    st.image("images/dataset_demo/6.jpg", caption="Realistic road conditions", use_container_width=True)
with col3:
    st.image("images/dataset_demo/4.jpg", caption="Potential false positives", use_container_width=True)

st.markdown("""
We continue our curriculum using **YouTube dashcam videos** to expose the model to **realistic driving conditions**, including **multiple angles, glare**, and other challenging scenarios. Videos are first downloaded from the dashcam and then sampled at **1 frame every 3 seconds**, after which **bounding boxes are manually annotated**.

To address the behavior observed in our first model — recognizing any **rectangular object with text** as a plate — we expanded the dataset to include **over 120 images of potential false positives**, such as **signboards and plate-like text**. Besides, we tackle **catastrophic forgetting** by including **88 clear and diverse on-road plate images**, ensuring the model retains knowledge of **genuine plates** while learning to ignore misleading patterns.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Precision", value="0.868")
with col2:
    st.metric(label="Recall", value="0.946")
with col3:
    st.metric(label="mAP@50", value="0.953")
with col4:
    st.metric(label="mAP@50-95", value="0.623")

st.markdown("""
**Dataset and Training Summary**

• Dataset Size: 421 images  
• Train: 332 | Validation: 59 | Test: 30  
• Training: 100 epochs  
• Image Size: 1280  
• Elapsed Time: 0:58:14 (3494s)
""")
st.divider()
#____________________________Section 2________________________________________________________________________


#____________________________Section 3________________________________________________________________________
st.markdown(
    "<h4 style='margin-top:1px;'>3. Dataset and Training V3</h4>", 
    unsafe_allow_html=True
)
st.markdown(
    "View Dataset: [v3_rainynight_taxiev_supplimentary-geioz](https://app.roboflow.com/yolo-zmazg/v3_rainynight_taxiev_supplimentary-geioz/1)"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.image("images/dataset_demo/7.jpg", caption="Rainy night conditions", use_container_width=True)
with col2:
    st.image("images/dataset_demo/8.jpg", caption="Supplimentary day images", use_container_width=True)
with col3:
    st.image("images/dataset_demo/9.jpg", caption="White Plates (EV and Taxi)", use_container_width=True)

st.markdown("""
We further our curriculum by introducing **low-light night conditions** (e.g., poorly lit streets, headlights glare) and **rainy situations** (with reflections and water streaks), while still preserving **signboards and other plate-like text** as **false positives**. This dataset is primarily sourced from **dashcam videos recorded in Malaysian traffic conditions**.  

To help the model **retain previously learned knowledge of clear daytime plates**, we include **supplementary daytime images** from another YouTube dashcam video covering urban and highway scenarios.  

Finally, we observed that **taxis and EVs in Malaysia often use white plates**, which can be **harder for the model to detect in varied lighting**. To address this, we sourced additional images from **Google Images**, including **close-ups and on-road shots**, ensuring the model learns to recognize these challenging cases.      
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Precision", value="0.974")
with col2:
    st.metric(label="Recall", value="0.841")
with col3:
    st.metric(label="mAP@50", value="0.933")
with col4:
    st.metric(label="mAP@50-95", value="0.618")

st.markdown("""
**Dataset and Training Summary**

• Dataset Size: 150 images  
• Train: 104 | Validation: 21 | Test: 25  
• Training: 100 epochs  
• Image Size: 1280  
• Elapsed Time: 0:19:40 (1180.221s)
""")
st.divider()
#____________________________Section 3________________________________________________________________________


