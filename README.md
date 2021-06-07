# pose_judge

posenet-python-masterディレクトリ直下（posenetディレクトリと同階層）に配置して動作する。  

・webcam_getdata.py  
　カメラによるデータ取得終了後に２次元座標データを.npy(多次元配列保存用ファイル)に出力  
・rate_graf.py  
　.npyファイルを元に距離比率をグラフに描画  
・webcam_rate_graf.py  
　リアルタイムで距離比計算、データ取得終了後にグラフ描画（上記2つの統合機能を実装）  
