import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Preprocessing import *
from Fusion import *
from Classifier import *
from utils import *

import sys  
import os
from sklearn.decomposition import PCA

def scMFF_model(dir, model_name):
    ####        Step 0: Data Loaded                  ########################################################
    print('---------------------------------------------------------------')
    print('Step 0: Data loading in progress...')
    df = pd.read_csv(dir + 'expr.csv', index_col=0, header=0)

    CellType = pd.read_csv(dir + 'cluster.csv', header=None).iloc[:, 0].values
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(CellType)
    num_classes = len(np.unique(Y))

    print('Data loading complete.')  

    ####        Step 1: Data Pre-processing          ########################################################
    print('---------------------------------------------------------------')
    print('Step 1: Data preprocessing in progress...')
    df0 = preprocess_RNASeq(df)
    df0.to_csv(dir + 'expr_2000.csv')
    print('Data preprocessing complete.')

    ####        Step 2: Feature Extraction Module    ########################################################
    print('---------------------------------------------------------------')
    print('Step 2: Feature extraction in progress...')
    
    # a. Statistics-based Feature: expr_2000.csv
    feature1 = df0.T.values
    print('(1) Statistics-based Feature: expr_2000.csv \t', feature1.shape)
    
    # b. Information-based Feature: scPSD      
    sys.path.append('scMFF/psdPy') 
    from psdPy import psd
    X_scPSD = psd(df0.T) 
    np.savetxt(dir + 'scPSD_embedding.txt.gz', X_scPSD)
    feature2 = X_scPSD
    print('(2) Information-based Feature: scPSD_embedding.txt.gz \t', feature2.shape)
    
    # c. Matrix Factorization-based Feature: PCA
    pca = PCA(n_components=266) 
    X_PCA = pca.fit_transform(df0.T)
    np.savetxt(dir + 'PCA_embedding.txt.gz', X_PCA)
    feature3 = X_PCA
    print('(3) Matrix Factorization-based Feature: PCA_embedding.txt.gz \t', feature3.shape)
    
    # d. Deep Learningg-based Feature: scGNN
    sys.path.append('scMFF/') 
    dir_path1 = '/home/sunnan/BioData/Transcriptome_classification/scMFF/scMFF/Example_data/'
    preprocess_command = f"python -W ignore scGNN/PreprocessingscGNN.py --datasetName expr.csv --datasetDir {dir_path1} --LTMGDir {dir_path1} --filetype CSV --geneSelectnum 2000"    
    os.system(preprocess_command)
    dir_path2 = '/home/sunnan/BioData/Transcriptome_classification/scMFF/scMFF/'
    scgnn_command = f"python -W ignore scGNN/scGNN.py --datasetName Example_data --datasetDir {dir_path2} --outputDir {dir_path1} --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode --nonsparseMode"
    os.system(scgnn_command)
    feature4 = pd.read_csv(dir + 'scGNN_embedding.csv', index_col=0, header=0).values
    print('(4) Deep Learningg-based Feature: scGNN_embedding.csv \t', feature4.shape)

    print('Feature extraction complete.')


    ####        Step 3: Cross Fusion Mudule          ########################################################
    print('---------------------------------------------------------------')
    print('Step 3: Cross Fusion in progress...')
    [n_sample, n_feature] = feature1.shape
    Fusion_sum, Fusion_hadamard, Fusion_attention = [], [], []
    Fused_moe, Fused_residual, Fused_transformer = [], [], []

    i = 0
    max_dim = max(len(feature1[i, :]), len(feature2[i, :]), len(feature3[i, :]), len(feature4[i, :]))
    output_dim = min(len(feature1[i, :]), len(feature2[i, :]), len(feature3[i, :]), len(feature4[i, :]))
   
    moefusion = MoEFusion(max_dim * 4, output_dim)
    residualfusion = ResidualFusion(max_dim, output_dim)
    transformerfusion = TransformerConcatFusion(max_dim, output_dim)
    
    for i in range(n_sample):
        print('The '+str(i+1)+'-th cell embedding')
        weights = [1/4, 1/4, 1/4, 1/4]
        fusion_sum = weighted_sum_fusion([feature1[i,:], feature2[i,:], feature3[i,:], feature4[i,:]], weights)
        fusion_hadamard = hadamard_product_fusion([feature1[i,:], feature2[i,:], feature3[i,:], feature4[i,:]])
        
        features = [torch.tensor(feature1[i,:], dtype=torch.float32), torch.tensor(feature2[i,:], dtype=torch.float32), torch.tensor(feature3[i,:], dtype=torch.float32), torch.tensor(feature4[i,:], dtype=torch.float32)]
        labels = torch.tensor(Y[i])    
        AttentionFusion_model = train_AttentionFusion(features, labels, num_classes)
        fusion_attention,logits = AttentionFusion_model(features)

        Fusion_sum.append(fusion_sum)
        Fusion_hadamard.append(fusion_hadamard)
        Fusion_attention.append(fusion_attention.detach().cpu().numpy())

        f1 = zero_pad_feature(feature1[i, :],max_dim)
        f2 = zero_pad_feature(feature2[i, :],max_dim)
        f3 = zero_pad_feature(feature3[i, :],max_dim)
        f4 = zero_pad_feature(feature4[i, :],max_dim) 

        fused_moe = moefusion(f1, f2, f3, f4)
        fused_residual = residualfusion(f1, f2, f3, f4)
        fused_transformer = transformerfusion(f1, f2, f3, f4)  

        Fused_moe.append(fused_moe)
        Fused_residual.append(fused_residual)
        Fused_transformer.append(fused_transformer)

    np.savetxt(dir + 'Fusion_sum-expr-scpsd-scgnn-pca.txt.gz', Fusion_sum)
    np.savetxt(dir + 'Fusion_hadamard-expr-scpsd-scgnn-pca.txt.gz', Fusion_hadamard)
    np.savetxt(dir + 'Fusion_attention-expr-scpsd-scgnn-pca.txt.gz', Fusion_attention)

    np.savetxt(dir + 'Fusion_moe-expr-scpsd-scgnn-pca.txt.gz', np.array([fused_moe.detach().numpy().flatten() for fused_moe in Fused_moe]))
    np.savetxt(dir + 'Fusion_residual-expr-scpsd-scgnn-pca.txt.gz', np.array([fused_residual.detach().numpy().flatten() for fused_residual in Fused_residual]))
    np.savetxt(dir + 'Fusion_transformer-expr-scpsd-scgnn-pca.txt.gz', np.array([fused_transformer.detach().numpy().flatten() for fused_transformer in Fused_transformer]))
    
    feature5 = Fusion_sum
    feature6 = Fusion_hadamard
    feature7 = Fusion_attention
    feature8 = np.array([fused_moe.detach().numpy().flatten() for fused_moe in Fused_moe])
    feature9 = np.array([fused_residual.detach().numpy().flatten() for fused_residual in Fused_residual])
    feature10 = np.array([fused_transformer.detach().numpy().flatten() for fused_transformer in Fused_transformer]) 

    print('Cross Fusion complete.')

    ####        Step 4: Classification Module        ########################################################
    unique_classes, counts = np.unique(Y, return_counts=True)
    min_count = 3
    for cls in unique_classes:
        if counts[cls] < min_count:
            Y[Y == cls] = max(unique_classes) + 1 
    new_label_encoder = LabelEncoder()
    y = new_label_encoder.fit_transform(Y)

    model = get_model(model_name)
    results1 = cross_validation_model(feature1, y, model)
    results2 = cross_validation_model(feature2, y, model)
    results3 = cross_validation_model(feature3, y, model)
    results4 = cross_validation_model(feature4, y, model)
    results5 = cross_validation_model(feature5, y, model)
    results6 = cross_validation_model(feature6, y, model)
    results7 = cross_validation_model(feature7, y, model)
    results8 = cross_validation_model(feature8, y, model)
    results9 = cross_validation_model(feature9, y, model)
    results10 = cross_validation_model(feature10, y, model)

    df = pd.DataFrame({
        'feature1': results1,
        'feature2': results2,
        'feature3': results3,
        'feature4': results4,
        'feature5': results5,
        'feature6': results6,
        'feature7': results7,
        'feature8': results8,
        'feature9': results9,
        'feature10': results10
    })

    df.to_excel(dir + 'results.xlsx')

    return df
