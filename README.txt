###################################################### brief introduction of important files ######################################################

- polyvore-images/ 		-- the directory of outfit image, we give a small sample of an outfit (original dataset on 							Google Drive)
	/[outfitid]/0.jpg 	-- the whole outfit performance
				1.jpg   the first items
				2.jpg   the second items

- polyvore_image_vectors/	
	/[outfitid]\_[itemid].json

- polyvore_text_vec/	the vector of all items by Word2Vec

- category_id.txt	the id to category name index for all items
 
- data_subset.ipynb	the file to subset 10% of the data

- image_embedding_using_VIT.ipynb	the file is used to create image embeddings using pretrained Vision Transformer model from huggingface
  
###################################################### how to prepare for our model training ######################################################
Zero:		    Change all the folder paths in the files to your corresponding ones

First:              download and unzip the polyvore-images from Google Drive [here](https://drive.google.com/drive/folders/0B4Eo9mft9jwoVDNEWlhEbUNUSE0)  

Second:             Click [here](https://drive.google.com/open?id=1ibYEw0H9L9O9OLbxCiAlcZkt_IYuwKfd) for the extracted feature using inception model.

Third:              download and unzip the detail informations of polyvore outfit.  [filtered verison](https://drive.google.com/open?id=1ibYEw0H9L9O9OLbxCiAlcZkt_IYuwKfd) Contains training data, test data, category-mapping and category-item mapping.

Fourth:		    run "onehot_embedding.py" to create the textual features (The rest of the pre-processing was already done by t#he authors)

#############################################################3 running the code: NGNN ##############################################################

Run main_multi_modal.py, this script trains and saves the NGNN model.

############################################################# running the code: HGNN (OCPHN) ########################################################### 
Run main_training.ipynb, this script trains and saves the HGNN model.







