"""
人脸数据库管理模块
负责存储和管理已注册人员的面部编码与姓名
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path


class FaceDatabase:
    """人脸数据库，管理已注册人员"""

    def __init__(self, db_path="face_db.json", encoding_path="encodings.pkl"):
        self.db_path = Path(db_path)
        self.encoding_path = Path(encoding_path)
        self.persons = []          # 人员列表：[{name, image_path}, ...]
        self.encodings = []        # 面部编码列表
        self.load()

    def add_person(self, name: str, image_path: str, encoding: np.ndarray):
        """添加人员记录"""
        # 检查是否已存在同名
        for person in self.persons:
            if person["name"] == name:
                return False, f"人员 '{name}' 已存在"

        person = {
            "name": name,
            "image_path": image_path,
        }
        self.persons.append(person)
        self.encodings.append(encoding)
        self.save()
        return True, f"已添加: {name}"

    def remove_person(self, name: str):
        """移除单个人员"""
        for i, person in enumerate(self.persons):
            if person["name"] == name:
                img_path = Path(person["image_path"])
                if img_path.exists():
                    img_path.unlink(missing_ok=True)
                del self.persons[i]
                del self.encodings[i]
                self.save()
                return True, f"已移除: {name}"
        return False, f"未找到: {name}"

    def remove_persons(self, names: list):
        """批量移除人员"""
        removed = []
        not_found = []
        to_remove_indices = []

        for name in names:
            found = False
            for i, person in enumerate(self.persons):
                if person["name"] == name:
                    img_path = Path(person["image_path"])
                    if img_path.exists():
                        img_path.unlink(missing_ok=True)
                    to_remove_indices.append(i)
                    removed.append(name)
                    found = True
                    break
            if not found:
                not_found.append(name)

        # 从后往前删除，避免索引漂移
        for i in sorted(to_remove_indices, reverse=True):
            del self.persons[i]
            del self.encodings[i]

        self.save()
        return removed, not_found

    def get_names(self) -> list:
        """获取所有已注册人员姓名"""
        return [p["name"] for p in self.persons]

    def get_encodings_and_names(self) -> tuple:
        """获取编码和姓名列表"""
        return self.encodings, [p["name"] for p in self.persons]

    def save(self):
        """持久化存储"""
        data = {
            "persons": self.persons,
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        with open(self.encoding_path, "wb") as f:
            pickle.dump(self.encodings, f)

    def load(self):
        """加载数据库"""
        if self.db_path.exists():
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.persons = data.get("persons", [])

        if self.encoding_path.exists():
            with open(self.encoding_path, "rb") as f:
                self.encodings = pickle.load(f)

    def clear(self):
        """清空数据库"""
        for person in self.persons:
            img_path = Path(person["image_path"])
            if img_path.exists():
                img_path.unlink(missing_ok=True)

        self.persons = []
        self.encodings = []
        if self.db_path.exists():
            self.db_path.unlink()
        if self.encoding_path.exists():
            self.encoding_path.unlink()