import modelbit, sys
from typing import *
from io import open
from _io import BytesIO
from keras.src.utils.image_utils import img_to_array
from keras.src.applications.resnet import preprocess_input
from tensorflow import keras
import base64
import PIL.Image as Image
import numpy as np

reference = modelbit.load_value("data/reference.pkl") # {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy', 4: 'Cherry_(including_sour)___Powdery_mildew', 5: 'Cherry_(including_sour)___healthy', 6: 'Corn_(...

loaded_model = keras.models.load_model('data/loaded_model.h5')

# main function
def prediction(base64_img):
    img_data = base64.b64decode(base64_img)
    img = Image.open(BytesIO(img_data))

        # Resize and preprocess the image
    img = img.resize((256, 256))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

        # Make prediction using loaded model
    pred = np.argmax(loaded_model.predict(img_array))
    
    # Make prediction using loaded model
    
    if (reference[pred] == "Apple___Apple_scab"):
        reference[pred] = "Crop: Apple, Disease: Apple Scab, About Disease: Apple scab is a common disease of plants in the rose family that is caused by the ascomycete fungus Venturia inaequalis."
    if (reference[pred] == "Apple___Black_rot"):
        reference[pred] = "Crop: Apple, Disease: Black Rot, About Disease: Black rot is a fungal disease, caused by Diplodia seriata on apple. This disease can cause defoliation, fruit rot, and limb cankers. Leaf symptoms start appearing about one to three weeks after petal fall. Initially, the infections start as tiny purple spots., Cure: Apply fungicides and maintain good orchard hygiene."
    if (reference[pred] == "Apple___healthy"):
        reference[pred] = "Crop: Apple, Disease: Healthy, About Disease: No Disease Found."
    if (reference[pred] == "Apple___Cedar_apple_rust"):
        reference[pred] = "Crop: Apple, Disease: Cedar Apple Rust, About Disease: Large infestations of this rust can reduce yield on apples, blemish the fruit, and lead to weakening and death of redcedar. This particular rust produces four kinds of spores: basidiospores, teliospores, spermatia, and aeciospores."
    if (reference[pred] == "Cherry_(including_sour)___healthy"):
        reference[pred] = "Crop: Cherry, Disease: Healhty, About Disease: No Disease Found."
    if (reference[pred] == "Cherry_(including_sour)___Powdery_mildew"):
        reference[pred] = "Crop: Cherry, Disease: Powdery mildew, About Disease: Powdery mildew of sweet and sour cherry is caused by Podosphaera clandestina, an obligate biotrophic fungus. Mid- and late-season sweet cherry (Prunus avium) cultivars are commonly affected, rendering them unmarketable due to the covering of white fungal growth on the cherry surface."
    if (reference[pred] == "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"):
        reference[pred] = "Crop: Corn (Maize), Disease: Cercospora leaf spot or Gray leaf spot, About Disease: Grey Leaf Spot is caused by a fungal pathogen (Pyricularia grisea) that readily infects and kills leaf blades. Leaf infections can progress into the crown area, resulting in death of individual plants. Moderate outbreaks of gray leaf spot result in clusters of thin, off-colored turf."
    if (reference[pred] == "Corn_(maize)___Northern_Leaf_Blight"):
        reference[pred] = "Crop: Corn (Maize), Disease: Northern Leaf Blight, About Disease: Northern corn leaf blight is caused by the fungus Setosphaeria turcica. Symptoms usually appear first on the lower leaves."
    if (reference[pred] == 'Corn_(maize)___Common_rust_'):
        reference[pred] = f"Crop: Corn (Maize), Disease: Common Rust on Crop, About Disease: Common rust produces rust-colored to dark brown, elongated pustules on both leaf surfaces. The pustules contain rust spores that are cinnamon brown in color. Pustules darken as they age."
    if (reference[pred] == 'Corn_(maize)___healthy'):
        reference[pred] = f"Crop: Corn (Maize), Disease: Healthy, About Disease: No Disease"
    if (reference[pred] == 'Grape___Black_rot'):
        reference[pred] = f"Crop: Grape, Disease: Black Rot, About Disease: Grape black rot is a fungal disease caused by an ascomycetous fungus, Guignardia bidwellii, that attacks grape vines during hot and humid weather."
    if (reference[pred] == 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'):
        reference[pred] = f"Crop: Grape, Disease: Leaf blight (Isariopsis Leaf Spot), About Disease: Initially lesions are dull red to brown in color turn black later. If disease is severe this lesions may coalesce. On berries we can see symptom similar to black rot but the entire clusters will collapse."
    if (reference[pred] == 'Grape___Esca_(Black_Measles)'):
        reference[pred] = f"Crop: Grape, Disease: Esca (Black Measles), About Disease: One of the common fungal diseases is Esca (Black Measles) which is found in the Grape Plants and can be easily identified as brown streaking lesions on any part of the leaf. The affected leaves can dry off completely and fall off from the plant prematurely which eventually results in death of the plant."
    if (reference[pred] == 'Grape___healthy'):
        reference[pred] = f"Crop: Grape, Disease: Healthy, About Disease: No Disease"
    if (reference[pred] == 'Orange___Haunglongbing_(Citrus_greening)'):
        reference[pred] = f"Crop: Orange, Disease: Haunglongbing (Citrus greening), About Disease: Huanglongbing is a bacterial disease that attacks the vascular system of plants. Once infected, there is no cure for the disease, and in areas where the disease is endemic, citrus trees decline and die within a few years."
    if (reference[pred] == 'Peach___Bacterial_spot'):
        reference[pred] = f"Crop: Peach, Disease: Bacterial Spot, About Disease: Peach scab, caused by the fungus Cladosporium carpophilum, is most often a problem in warm, wet weather following shuck-split and is also known as ink spot. Although the fungus infects leaves and twigs, disease symptoms are most often observed on the fruit."
    if (reference[pred] == 'Peach___healthy'):
        reference[pred] = f"Crop: Peach, Disease: Healthy, About Disease: No Disease."
    if (reference[pred] == 'Pepper,_bell___Bacterial_spot'):
        reference[pred] = f"Crop: Pepper, Disease: bell Bacterial spot, About Disease: The bacterium is carried within seed, or in solanaceous weeds and crop debris. Tomatoes are also susceptible. Disease development is favored by high nighttime temperatures and high moisture and may be arrested during prolonged dry spells."
    if (reference[pred] == 'Pepper,_bell___healthy'):
        reference[pred] = f"Crop: Pepper, Disease: Healthy, About Disease: No Disease."
    if (reference[pred] == 'Potato___Early_blight'):
        reference[pred] = f"Crop: Potato, Disease: Early Blight, About Disease: Early blight of potato is caused by the fungus, Alternaria solani, which can cause disease in potato, tomato, other members of the potato family, and some mustards. This disease, also known as target spot, rarely affects young, vigorously growing plants."
    if (reference[pred] == 'Potato___Late_blight'):
        reference[pred] = f"Crop: Potato, Disease: Late Blight, About Disease: Late blight is caused by the fungal-like oomycete pathogen Phytophthora infestans."
    if (reference[pred] == 'Potato___healthy'):
        reference[pred] = f"Crop: Potato, Disease: Healthy, About Disease: No Disease."
    if (reference[pred] == 'Squash___Powdery_mildew'):
        reference[pred] = f"Crop: Squash, Disease: Powdery Mildew, About Disease: Powdery mildew symptoms appear on leaves as white, powdery, circular to irregularly shaped patches. Mildew lesions are apparent on both sides of leaves as dusty, grayish brown deposits."
    if (reference[pred] == 'Strawberry___Leaf_scorch'):
        reference[pred] = f"Crop: Strawberry, Disease: Leaf Scorch, About Disease: Leaf scorch is caused by the fungus Diplocarpon earliana. Symptoms can be mistaken for common leaf spot, caused by M. fragariae."
    if (reference[pred] == 'Strawberry___healthy'):
        reference[pred] = f"Crop: Strawberry, Disease: Healthy, About Disease: No Disease."
    if (reference[pred] == 'Tomato___Bacterial_spot'):
        reference[pred] = f"Crop: Tomato, Disease: Bacterial Spot, About Disease: The disease affects all above ground parts of a tomato plant, including leaves, stems, and fruit. In rainy and wet weather conditions, the disease can cause early defoliation and fruit spotting, which results in reduced yield and non-marketable fruit., Cure: Use copper-based fungicides and practice crop rotation."
    if (reference[pred] == 'Tomato___healthy'):
        reference[pred] = f"Crop: Tomato, Disease: Healthy, About Disease: No Disease."
    if (reference[pred] == 'Tomato___Late_blight'):
        reference[pred] = f"Crop: Tomato, Disease: Late Blight, About Disease: Late blight is a potentially serious disease of potato and tomato and is caused by the water mold pathogen Phytophthora infestans. Late blight is especially damaging during cool, wet weather."
    if (reference[pred] == 'Tomato___Early_blight'):
        reference[pred] = f"Crop: Tomato, Disease: Early Blight, About Disease: Early blight, caused by Alternaria solani, is a common fungal disease of tomatoes grown in fields, greenhouses, and high tunnels. In warm, rainy and wet weather, epidemics of this disease can cause severe defoliation, yield loss, and poor fruit quality. The fungus also infects potato."
    if (reference[pred] == 'Tomato___Leaf_Mold'):
        reference[pred] = f"Crop: Tomato, Disease: Leaf Mold, About Disease: Tomato leaf mold is caused by a fungal pathogen called Passalora fulva (syn. Cladosporium fulvum). It is an ascomycete fungus that lives on living tomato leaves. The fungus produces conidia that infect the lower surfaces of leaves."
    if (reference[pred] == 'Tomato___Septoria_leaf_spot'):
        reference[pred] = f"Crop: Tomato, Disease: Septoria Leaf Spot, About Disease: Septoria leaf spot is caused by the fungus Septoria lycopersici. This fungus can attack tomatoes at any stage of development, but symptoms usually first appear on the older, lower leaves and stems when plants are setting fruit."
    if (reference[pred] == 'Tomato___Spider_mites Two-spotted_spider_mite'):
        reference[pred] = f"Crop: Tomato, Disease: Spider Mites, About Disease: Tomato red spider mite feeding causes whitening or yellowing of leaves, which then dry out and eventually fall off. In the case of severe attacks, plant damage progresses very quickly, and hosts may die within 3-5 weeks, if no management actions are taken."
    if (reference[pred] == 'Tomato___Target_Spot'):
        reference[pred] = f"Crop: Tomato, Disease: Target Spot, About Disease: Target spot of tomato is caused by the fungal pathogen Corynespora cassiicola. 1 The disease occurs on field-grown tomatoes in tropical and subtropical regions of the world."
    if (reference[pred] == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'):
        reference[pred] = f"Crop: Tomato, Disease: Yellow Leaf Curl Virus, About Disease: Tomato yellow leaf curl virus is a DNA virus from the genus Begomovirus and the family Geminiviridae. TYLCV causes the most destructive disease of tomato, and it can be found in tropical and subtropical regions causing severe economic losses."
    if (reference[pred] == 'Tomato___Tomato_mosaic_virus'):
        reference[pred] = f"Crop: Tomato, Disease: Mosaic Virus, About Disease: The mosaic virus is a parasite that destroys plants, gardens, and crops down to their molecular level. Once a plant contracts the mosaic virus, the infected plant can then spread the virus to other plants and even affect an entire harvest if left untreated."
    
    
    print(f"{reference[pred]}")

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = prediction(...)
#   print(result)