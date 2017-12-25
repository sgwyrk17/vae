# vae   

## Required   
tensorflow 0.8.0   
glob(画像のパスを取得するため)   
numpy   
scipy   
opencv2   

## Usage   
### train   
`python main.py --train True`   

### test
`python main.py`   

## そもそも    
mnistでvae   
モデルはエンコーダー(conv2層・全結合層)、デコーダー(全結合層・deconv2層)   
