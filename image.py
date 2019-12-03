import cv2
import numpy as np

def img_read(file):
    #画像の読み込み
    img = cv2.imread(file)
    return img
def camera_read():
    path =None

    cap = cv2.VideoCapture(0)

    while(True):
        # フレームをキャプチャする
        ret, frame = cap.read()
        # 画面に表示する
        cv2.imshow('frame',frame)
        # キーボード入力待ち
        key = cv2.waitKey(1) & 0xFF

        # qが押された場合は終了する
        if key == ord('q'):
            break
        # sが押された場合は保存する
        if key == ord('s'):
            path = "photo.jpg"
            cv2.imwrite(path,frame)
    # キャプチャの後始末と，ウィンドウをすべて消す
    cap.release()
    cv2.destroyAllWindows()
    return path
   


def sub_color(src, K):
    # 次元数を1落とす
    Z = src.reshape((-1,3))
    # float32型に変換
    Z = np.float32(Z)
    # 基準の定義
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K-means法で減色
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # UINT8に変換
    center = np.uint8(center)
    res = center[label.flatten()]
    # 配列の次元数と入力画像と同じに戻す
    return res.reshape((src.shape))

def anime_filter(img, K):
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # ぼかしでノイズ低減
    edge = cv2.blur(gray, (3, 3))
    # Cannyアルゴリズムで輪郭抽出
    edge = cv2.Canny(edge, 50, 150, apertureSize=3)
    # 輪郭画像をRGB色空間に変換
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    # 画像の領域分割
    img = sub_color(img, K)
    # 差分を返す
    return cv2.subtract(img, edge)



while True:
    n = input('image or camera?: ')
    if n=='image':
        x = input('file name?')
        img = img_read(x)
        break
    elif n=='camera':
        print('q:quit,s:save')
        x = camera_read()
        img =img_read(x)
        break
    else:
        continue


#顔の検出
cas = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
faces = cas.detectMultiScale(img,minSize=(100,100))
#検出したのをかきこんできりとる
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    img = img[y:y+h,x:x+w]
#画像のリサイズ
img = cv2.resize(img,(300,300))
img=anime_filter(img,30)
#画像を表示
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()