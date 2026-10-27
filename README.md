# wearable-slam-haptic

JetsonとMID-360で、点群処理と振動通知を少しずつ作っています。

ROSのノードでLiDARのトピックを受信し、50 cm以内の点を除いて、残った点群を別のトピックに送るところまでできました。実際のLiDARでも受信できました。

入力は`/livox/lidar`、出力は`/haptic_learning/filtered_points`です。処理は`filter_logic.py`、ROSの受信と送信は`lidar_node.py`にあります。まだ身体と障害物の区別はしていません。

次は、50 cmより遠い点の中から一番近い点を探し、その座標と距離をトピックで送ってみます。

ROS Noeticと、ビルド済みの`livox_ros_driver2`が必要です。今のJetsonでは、ドライバを起動してから別の端末で実行します。

```bash
bash run.sh _input_topic:=/livox/lidar
```

`mock_lidar.py`は毎秒4点を送る確認用です。模擬入力を使うときは、受信側を次のように起動します。

```bash
bash run.sh _input_topic:=/haptic_learning/mock_lidar
```
