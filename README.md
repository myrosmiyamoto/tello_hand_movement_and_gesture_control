# Tello ドローンを制御するプログラムの説明

## 概要
このプログラムは、**DJI Tello** という小型ドローンをPythonを使って制御するものです。プログラムを実行すると、

- **カメラ映像を取得して表示**
- **手の動きを検出してドローンを制御**
- **キーボード入力でドローンを操作**

できるようになります。

## 必要なもの
このプログラムを動かすには、以下のものが必要です。

1. **DJI Tello ドローン**
2. **Wi-Fi 接続**（Tello の Wi-Fi に接続）
3. **Python3 がインストールされたPC**
4. **必要なライブラリのインストール**

### ライブラリのインストール
このプログラムを実行する前に、以下のライブラリをインストールしてください。

```bash
pip install djitellopy opencv-python mediapipe numpy
```

### キーボードでドローンを操作
キーボードのキーを押すと、ドローンを自由に動かせます。

| キー | 動作 |
|------|------|
| `t`  | 離陸 |
| `l`  | 着陸 |
| `w`  | 前進 |
| `s`  | 後退 |
| `a`  | 左へ移動 |
| `d`  | 右へ移動 |
| `e`  | 時計回りに旋回 |
| `q`  | 反時計回りに旋回 |
| `r`  | 上昇 |
| `f`  | 下降 |
| `p`  | バッテリー残量などの情報を表示 |
| `1`  | 自動モード ON |
| `0`  | 自動モード OFF |

`ESC`キーを押すか、ウィンドウを閉じると **プログラムが終了** し、Tello も停止します。

## プログラムの実行方法
1. **Tello の Wi-Fi に接続する**（スマホアプリでなく、PCから接続）
1. **プログラムを実行**

```bash
python3 tello_hand_movement_and_gesture_control.py
```

1. **カメラ映像が表示され、操作可能に！**
1. **`t` キーの押してドローンを離陸させる**
1. **`1` キーの押して自動制御をONにする**
1. **ドローンのカメラに手を映すとドローンを制御できる**

- **ドローンのカメラの中心に手が来るようにドローンが追従する**
- **手を開き大きくするとドローンが離れる**
- **手を閉じ小さくするとドローンが近づいてくる**
- **人差し指と中指だけを開く（✌️） → ドローンが喜んで宙返りする**

## プログラムの説明
`_detect_hand_movement_and_gesture` メソッドが **手の動きを検出し、ドローン（Tello）を操作する** ための処理を行っています。
具体的には、カメラの映像から **手の位置や形（ジェスチャー）を認識し、それに応じてドローンを動かす** ことができます。
カメラの映像から手の位置や形（ジェスチャー）を認識し、それに応じてドローンを動かすプログラムの処理について説明します。


### (1) 映像をRGBに変換
```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
result = self.hands.process(rgb_frame)
```
カメラから取得した画像はBGR（青・緑・赤）の順になっていますが、**Mediapipe（手の検出ライブラリ）** はRGB（赤・緑・青）の順番で画像を処理します。
そのため、`cv2.cvtColor()` を使って色の順番を変換しています。

その後、`self.hands.process(rgb_frame)` で手の位置を検出します。
この関数が **「手のランドマーク（指や手の関節の位置）」** を見つけてくれます。

### (2) 手が映っているかチェック
```python
if result.multi_hand_landmarks:
```
もし `result.multi_hand_landmarks` にデータが入っていれば、**手が映っている** ことが確認できます。

### (3) 手の形を囲む四角形を描画
```python
for hand_landmarks in result.multi_hand_landmarks:
    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    for lm in hand_landmarks.landmark:
        h, w, _ = frame.shape
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
```
- `draw_landmarks()` で **手のランドマーク（指の関節など）** を映像上に描画します。
- 手の **一番左上と右下の座標** を求め、それを囲む四角形（緑色）を `cv2.rectangle()` で描画します。

### (4) ドローンを中央に移動させるための計算
```python
if self.is_automode:
    a = b = c = d = 0
    width = x_max - x_min
    height = y_max - y_min
    center_x = int(x_min + width / 2)
    center_y = int(y_min + height / 2)

    dx = 0.3 * (self.target_x - center_x)
    dy = 0.3 * (self.target_y - center_y)
    dw = 0.4 * (80 - width)
```
- `center_x`, `center_y` は手の **中心の座標**
- `dx` は **手のX座標が画面の中心（`self.target_x`）からどれくらいずれているか**
- `dy` は **手のY座標が画面の中心（`self.target_y`）からどれくらいずれているか**
- `dw` は **手の大きさが基準（80px）からどれくらい違うか**

**この値を元に、ドローンを動かすための制御信号を作成します。**

### (5) 不感帯（小さな動きを無視）
```python
d = 0.0 if abs(dx) < 20.0 else dx
b = 0.0 if abs(dw) < 10.0 else dw
c = 0.0 if abs(dy) < 30.0 else dy
```
**「微妙な動きには反応しない」** ようにするため、一定の範囲（20px, 10px, 30px）以内のずれは **0に設定** します。

### (6) 手の形（ジェスチャー）の認識
```python
fingers = []
for tip_id in [8, 12, 16, 20]:
    fingers.append(hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y)
```
- `8, 12, 16, 20` はそれぞれ **人差し指・中指・薬指・小指の先端**
- その先端が第二関節よりも **上にあれば「指が立っている」** と判断します。

### (7) ジェスチャーの判定
```python
if all(fingers[1:]):  # すべての指が開いている
    gesture = "paper"
elif not any(fingers):  # すべての指が閉じている
    gesture = "rock"
elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
    gesture = "Scissors"
    if not self.is_flip:
        self.is_flip = True
        Thread(target=self._tello_control, args=('b_flip', )).start()
else:
    gesture = "nothing"
```
- **パー（paper）**: 指がすべて開いている
    - 特に処理は与えていない
- **グー（rock）**: 指がすべて閉じている
    - 特に処理は与えていない
- **チョキ（scissors）**: **人差し指と中指だけ開いている**
  - チョキをすると、`self.is_flip` が `True` になり、**ドローンが宙返り（フリップ）する**

### (8) 画面にジェスチャーを表示
```python
cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
```
検出した **手の形（グー・チョキ・パー）** を画面に表示します。

## まとめ
このプログラムを使えば、**Python でドローンを制御** できるようになります。

- **キーボードを使って自由に操縦する**
- **カメラ映像を取得する**
- **手の動きを使ってドローンを操作する**
    - **手の位置が画面の中心からどれだけズレているかでドローンが動く**
    - **特定のジェスチャーでドローンを動かす（例: チョキで宙返り）**
