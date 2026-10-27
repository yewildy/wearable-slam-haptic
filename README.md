# wearable-slam-haptic

JetsonとMID-360を使って、周囲の物体との距離に応じた振動通知を作りたいと考えています。まずはPythonで点群処理を一つずつ試しています。

今は、LiDARから50 cm以内の点を取り除くところまで書きました。まだ身体と障害物を区別する処理ではありません。次は実際のLiDARの点群で動きを確認します。

`filter_logic.py`が処理本体、`lidar_node.py`がROSの受信と送信です。`mock_lidar.py`は毎秒4点を送る確認用です。

ROS Noeticと、ビルド済みの`livox_ros_driver2`が必要です。`run.sh`は現在使っているJetsonのワークスペースを読み込みます。

```bash
# 4点の簡単な確認（ROS不要）
python3 filter_logic.py

# ROSの模擬入力で確認（別の端末でroscoreを起動しておく）
source /opt/ros/noetic/setup.bash
source ~/haptic_ws/devel/setup.bash
python3 mock_lidar.py

# もう一つの端末
bash run.sh _input_topic:=/haptic_learning/mock_lidar
```
