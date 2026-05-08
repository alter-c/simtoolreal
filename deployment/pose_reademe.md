## 在一个终端启动

```bash
cd /home/unitree/simtoolreal && uva
python /home/unitree/simtoolreal/deployment/current_pose_node.py
```

---

## 在另一个终端启动

```bash
cd /home/unitree/simtoolreal && uva
python /home/unitree/simtoolreal/deployment/goal_pose_node.py
```

---

### 说明

如果换物体，需要在 `goal_pose_node.py` 里面更改一下内容，轨迹存放在：

```
/home/unitree/simtoolreal/dextoolbench/trajectories/hezi/hezi
```

修改地方如下：

```python
object_category: str = "hezi"
object_name: str = "hezi"
task_name: str = "对应json文件"
```


