# wearable-slam-haptic

JetsonとMID-360で、点群処理と振動通知を少しずつ作っています。

ROSのノードでLiDARのトピックを受信し、50 cm以内の点を除いて、残った点群を別のトピックに送ります。今回は、残った点の中で一番近い距離も計算し、端末に表示できました。

今使っている処理は`filter_logic_closer_num.py`です。点群と最小距離を一緒に返し、`lidar_node.py`で二つの変数に受け取っています。前の`filter_logic.py`も残しています。

入力は`/livox/lidar`、点群の出力は`/haptic_learning/filtered_points`です。距離はまだ専用トピックには送っていません。身体と障害物の区別もしていません。

次は、距離が近いほど通知を強くする、簡単な線形の関係を試したいです。まだ実装していません。

ROS Noeticと、ビルド済みの`livox_ros_driver2`が必要です。ドライバ起動後、別の端末で実行します。

```bash
bash run.sh _input_topic:=/livox/lidar
```

模擬入力を使う場合：

```bash
bash run.sh _input_topic:=/haptic_learning/mock_lidar
```
