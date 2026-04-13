import yaml
import xml.etree.ElementTree as ET

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def make_joint(
    joint_name: str,
    parent_link: str,
    child_link: str,
    xyz: str = "0 0 0",
    rpy: str = "0 0 0",
    joint_type: str = "fixed",
) -> ET.Element:
    joint = ET.Element("joint", name=joint_name, type=joint_type)
    ET.SubElement(joint, "parent", link=parent_link)
    ET.SubElement(joint, "child", link=child_link)
    ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
    return joint

def merge_urdfs_side(g1_urdf, linkerhand_urdf, side="left"):
    G1 = ET.parse(g1_urdf)
    g1 = G1.getroot()
    hand = ET.parse(linkerhand_urdf).getroot()

    # clean
    for element in list(g1):
        name = element.attrib.get('name')
        if element.tag == "link" and name in config['G1_remove_links']:
            print('[INFO] Remove link', name)
            g1.remove(element)
        if element.tag == "joint" and name in config['G1_remove_joints']:
            print('[INFO] Remove joint', name)
            g1.remove(element)
    for element in hand:
        if element.tag in ["link", "joint"]:
            g1.append(element)

    # merge
    rpys = {
        "left": "-1.5707963267 0 -1.5707963267",   
        "right": "1.5707963267 0 1.5707963267",  
    }
    sd = "l" if side == "left" else "r"
    connecting_joint = make_joint(
        joint_name  = f"{side}_linkerhand_base_joint",
        parent_link = f"{side}_wrist_yaw_link", # G1
        child_link  = f"{sd}h_hand_base_link",  # Linkerhand
        xyz         = "0.0415 0 0",               
        rpy         = rpys[side], 
    )
    g1.append(connecting_joint)
    print(f"[INFO] Inserted connecting joint: {side}_wrist_yaw_link -> lh_hand_base_link")

    output = g1_urdf[:-5] + f"_with_linkerhand_{side}.urdf"
    G1.write(output)
    print(f"[INFO] Generated: {output}")

def merge_urdfs(g1_urdf, linkerhand_left_urdf, linkerhand_right_urdf):
    G1 = ET.parse(g1_urdf)
    g1 = G1.getroot()
    lhand = ET.parse(linkerhand_left_urdf).getroot()
    rhand = ET.parse(linkerhand_right_urdf).getroot() 

    # clean
    for element in list(g1):
        name = element.attrib.get('name')
        if element.tag == "link" and name in config['G1_remove_links']:
            print('[INFO] Remove link', name)
            g1.remove(element)
        if element.tag == "joint" and name in config['G1_remove_joints']:
            print('[INFO] Remove joint', name)
            g1.remove(element)
    for element in lhand:
        if element.tag in ["link", "joint"]:
            g1.append(element)
    for element in rhand:
        if element.tag in ["link", "joint"]:
            g1.append(element)

    # merge
    rpys = {
        "left": "-1.5707963267 0 -1.5707963267",   
        "right": "1.5707963267 0 1.5707963267",  
    }
    for side in ["left", "right"]:
        s = "l" if side == "left" else "r"
        connecting_joint = make_joint(
            joint_name  = f"{side}_linkerhand_base_joint",
            parent_link = f"{side}_wrist_yaw_link",   
            child_link  = f"{s}h_hand_base_link",   
            xyz         = "0.0415 0 0",           
            rpy         = rpys[side],  
        )
        g1.append(connecting_joint)
        print(f"[INFO] Inserted connecting joint: {side}_wrist_yaw_link -> lh_hand_base_link")

    output = g1_urdf[:-5] + f"_with_linkerhand.urdf"
    G1.write(output)
    print(f"[INFO] Generated: {output}")


if __name__ == "__main__":
    merge_urdfs(
        g1_urdf        = "g1_29dof.urdf",
        linkerhand_left_urdf = "linkerhand_o6_left.urdf",
        linkerhand_right_urdf = "linkerhand_o6_right.urdf",
    )