#!/usr/bin/env python3
"""
飞书文档集成 - Python SDK
支持查看、读取、写入飞书文档
"""

import os
import sys
import json
import requests
from typing import Optional, Dict, List, Any
from datetime import datetime
import time


class FeishuDocClient:
    """飞书文档客户端"""

    def __init__(self, app_id: str, app_secret: str, tenant_id: str):
        """
        初始化飞书文档客户端

        Args:
            app_id: 飞书应用ID
            app_secret: 飞书应用Secret
            tenant_id: 飞书租户ID
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self.tenant_id = tenant_id
        self.access_token = None
        self.token_expires = None
        self.base_url = "https://open.feishu.cn/open-apis"

        # 日志
        self.log_dir = os.path.expanduser("~/.feishu_sdk")
        os.makedirs(self.log_dir, exist_ok=True)

    def _log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(
            self.log_dir, f"feishu_{datetime.now().strftime('%Y%m%d')}.log"
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")

        print(f"[{level}] {message}")

    def _get_app_access_token(self) -> str:
        """获取应用访问令牌"""
        if self.access_token and time.time() < self.token_expires:
            return self.access_token

        url = f"{self.base_url}/auth/v3/app_access_token/internal"

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.app_secret}",
        }

        data = {"app_id": self.app_id, "app_secret": self.app_secret}

        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()

            result = response.json()

            if result.get("code") != 0:
                self._log(f"获取token失败: {result}", "ERROR")
                raise Exception(f"获取token失败: {result.get('msg', '未知错误')}")

            self.access_token = result["app_access_token"]
            self.token_expires = time.time() + 3600  # 1小时后过期

            self._log(f"成功获取飞书访问令牌", "SUCCESS")
            return self.access_token

        except Exception as e:
            self._log(f"获取token异常: {str(e)}", "ERROR")
            raise

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None,
    ) -> Any:
        """发送API请求"""

        access_token = self._get_app_access_token()

        url = f"{self.base_url}{endpoint}"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        if params:
            url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method == "POST":
                if files:
                    response = requests.post(
                        url, headers=headers, data=data, files=files, timeout=30
                    )
                else:
                    response = requests.post(
                        url, headers=headers, json=data, timeout=30
                    )
            elif method == "PATCH":
                response = requests.patch(url, headers=headers, json=data, timeout=30)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()
            return response.json()

        except Exception as e:
            self._log(f"API请求异常: {method} {endpoint} - {str(e)}", "ERROR")
            raise

    def list_documents(
        self, space_id: str, page_size: int = 50, page_token: str = ""
    ) -> Dict[str, Any]:
        """
        查看文档列表

        Args:
            space_id: 云空间ID
            page_size: 每页数量（默认50）
            page_token: 分页token（翻页）

        Returns:
            文档列表响应
        """
        self._log(f"查看文档列表 - space_id: {space_id}")

        endpoint = "/docx/v1/documents"

        data = {"space_id": space_id, "page_size": page_size}

        if page_token:
            data["page_token"] = page_token

        return self._make_request("POST", endpoint, data=data)

    def get_document(
        self, document_id: str, obj_type: str = "docx", obj_basic: bool = True
    ) -> Dict[str, Any]:
        """
        读取文档内容

        Args:
            document_id: 文档ID
            obj_type: 对象类型
            obj_basic: 是否返回基本信息

        Returns:
            文档响应
        """
        self._log(f"读取文档 - document_id: {document_id}, obj_type: {obj_type}")

        endpoint = f"/docx/v1/{obj_type}/{document_id}"

        params = {}
        if obj_basic:
            params["obj_basic"] = "true"

        return self._make_request("GET", endpoint, params=params)

    def create_document(
        self, space_id: str, title: str, content: str, folder_token: str = ""
    ) -> Dict[str, Any]:
        """
        创建新文档

        Args:
            space_id: 云空间ID
            title: 文档标题
            content: 文档内容（Base64编码）
            folder_token: 文件夹token（可选）

        Returns:
            创建响应
        """
        self._log(f"创建文档 - space_id: {space_id}, title: {title}")

        endpoint = "/docx/v1/documents"

        data = {"space_id": space_id, "title": title, "content": content}

        if folder_token:
            data["folder_token"] = folder_token

        return self._make_request("POST", endpoint, data=data)

    def update_document(
        self,
        document_id: str,
        obj_type: str = "docx",
        title: str = None,
        content: str = None,
    ) -> Dict[str, Any]:
        """
        更新文档内容

        Args:
            document_id: 文档ID
            obj_type: 对象类型
            title: 文档标题（可选）
            content: 文档内容（Base64编码，可选）

        Returns:
            更新响应
        """
        self._log(f"更新文档 - document_id: {document_id}, obj_type: {obj_type}")

        endpoint = f"/docx/v1/{obj_type}/{document_id}"

        data = {}
        if title:
            data["title"] = title
        if content:
            data["content"] = content

        return self._make_request("PATCH", endpoint, data=data)

    def delete_document(
        self, document_id: str, obj_type: str = "docx"
    ) -> Dict[str, Any]:
        """
        删除文档

        Args:
            document_id: 文档ID
            obj_type: 对象类型

        Returns:
            删除响应
        """
        self._log(f"删除文档 - document_id: {document_id}, obj_type: {obj_type}")

        endpoint = f"/docx/v1/{obj_type}/{document_id}"

        return self._make_request("DELETE", endpoint)


def base64_encode_file(file_path: str) -> str:
    """
    Base64编码文件内容

    Args:
        file_path: 文件路径

    Returns:
        Base64编码的字符串
    """
    import base64

    with open(file_path, "rb") as f:
        file_content = f.read()

    return base64.b64encode(file_content).decode("utf-8")


def interactive_menu():
    """交互式菜单"""

    print("=" * 80)
    print("  飞书文档集成 - Python SDK")
    print("=" * 80)
    print()

    print("请选择功能:")
    print()
    print("  1. 查看文档列表")
    print("  2. 读取文档内容")
    print("  3. 创建新文档")
    print("  4. 更新文档内容")
    print("  5. 搜索文档")
    print("  6. 删除文档")
    print("  7. 查看日志")
    print("  0. 退出")
    print()

    return input("输入选项 (0-7): ")


def main():
    """主函数"""

    print()
    print("=" * 80)
    print("  飞书文档集成 - Python SDK")
    print("=" * 80)
    print()
    print("功能:")
    print("  ✅ 查看文档列表")
    print("  ✅ 读取文档内容")
    print("  ✅ 写入文档内容")
    print("  ✅ 创建新文档")
    print("  ✅ 更新文档内容")
    print("  ✅ 删除文档")
    print("  ✅ 搜索文档")
    print()
    print("=" * 80)
    print()

    # 配置信息
    config_file = os.path.expanduser("~/.feishu_sdk/config.json")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)

    # 检查配置
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        print(f"📄 找到配置文件: {config_file}")
        print(f"  App ID: {config.get('app_id', '未设置')}")
        print(f"  App Secret: {'*' * len(config.get('app_secret', ''))}")
        print(f"  Tenant ID: {config.get('tenant_id', '未设置')}")
        print()

        use_existing = input("使用现有配置? (y/n): ").strip().lower()

        if use_existing != "y":
            config = {}
    else:
        config = {}

    if (
        not config.get("app_id")
        or not config.get("app_secret")
        or not config.get("tenant_id")
    ):
        print("请配置飞书应用凭证:")
        print()
        config["app_id"] = input("App ID: ").strip()
        config["app_secret"] = input("App Secret: ").strip()
        config["tenant_id"] = input("Tenant ID: ").strip()
        print()

        # 保存配置
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✅ 配置已保存到: {config_file}")
        print()

    # 创建客户端
    try:
        client = FeishuDocClient(
            app_id=config["app_id"],
            app_secret=config["app_secret"],
            tenant_id=config["tenant_id"],
        )

        print("✅ 飞书文档客户端创建成功")
        print()

    except Exception as e:
        print(f"❌ 创建客户端失败: {str(e)}")
        print()
        print("💡 提示:")
        print("  1. 请确保配置信息正确")
        print("  2. 请确保已安装requests库: pip install requests")
        print("  3. 请检查网络连接")
        print()
        return

    # 交互式菜单
    while True:
        choice = interactive_menu()

        if choice == "0":
            print("退出...")
            break

        elif choice == "1":
            # 查看文档列表
            space_id = input("输入云空间ID: ").strip()
            page_size = input("每页数量 (默认50): ").strip() or "50"

            try:
                result = client.list_documents(
                    space_id=space_id, page_size=int(page_size)
                )

                if result.get("code") == 0:
                    items = result.get("data", {}).get("items", [])
                    print(f"✅ 找到 {len(items)} 个文档")

                    for i, item in enumerate(items[:10], 1):
                        doc_id = item.get("document_id", "N/A")
                        title = item.get("title", "N/A")
                        print(f"  {i}. {doc_id} - {title}")

                    if len(items) > 10:
                        print(f"  ... 还有 {len(items) - 10} 个文档")

                else:
                    print(f"❌ 失败: {result.get('msg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "2":
            # 读取文档
            document_id = input("输入文档ID: ").strip()
            obj_type = input("对象类型 (默认docx): ").strip() or "docx"

            try:
                result = client.get_document(document_id=document_id, obj_type=obj_type)

                if result.get("code") == 0:
                    print(f"✅ 成功读取文档: {document_id}")
                    print(f"   标题: {result.get('data', {}).get('title', 'N/A')}")

                    # 保存到文件
                    file_path = f"feishu_{document_id}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    print(f"   已保存到: {file_path}")

                else:
                    print(f"❌ 失败: {result.get('msg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "3":
            # 创建文档
            space_id = input("输入云空间ID: ").strip()
            title = input("输入文档标题: ").strip()
            file_path = input("输入文件路径 (本地文件): ").strip()
            folder_token = input("文件夹token (可选，回车跳过): ").strip()

            # Base64编码文件
            try:
                content = base64_encode_file(file_path)
            except Exception as e:
                print(f"❌ 文件编码失败: {str(e)}")
                continue

            try:
                result = client.create_document(
                    space_id=space_id,
                    title=title,
                    content=content,
                    folder_token=folder_token,
                )

                if result.get("code") == 0:
                    doc_id = (
                        result.get("data", {})
                        .get("document", {})
                        .get("document_id", "N/A")
                    )
                    print(f"✅ 成功创建文档: {doc_id}")
                else:
                    print(f"❌ 失败: {result.get('msg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "4":
            # 更新文档
            document_id = input("输入文档ID: ").strip()
            title = input("新标题 (回车跳过): ").strip() or None
            file_path = input("新文件路径 (回车跳过): ").strip() or None

            content = None
            if file_path:
                try:
                    content = base64_encode_file(file_path)
                except Exception as e:
                    print(f"❌ 文件编码失败: {str(e)}")
                    continue

            try:
                result = client.update_document(
                    document_id=document_id, title=title, content=content
                )

                if result.get("code") == 0:
                    print(f"✅ 成功更新文档: {document_id}")
                else:
                    print(f"❌ 失败: {result.get('msg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "5":
            # 搜索文档
            space_id = input("输入云空间ID: ").strip()
            query = input("搜索关键词: ").strip()
            page_size = input("每页数量 (默认50): ").strip() or "50"

            try:
                result = client.search_documents(
                    space_id=space_id, query=query, page_size=int(page_size)
                )

                if result.get("code") == 0:
                    items = result.get("data", {}).get("items", [])
                    print(f"✅ 找到 {len(items)} 个匹配文档")

                    for i, item in enumerate(items[:10], 1):
                        doc_id = item.get("document_id", "N/A")
                        title = item.get("title", "N/A")
                        print(f"  {i}. {doc_id} - {title}")

                    if len(items) > 10:
                        print(f"  ... 还有 {len(items) - 10} 个文档")

                else:
                    print(f"❌ 失败: {result.get('msg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "6":
            # 删除文档
            document_id = input("输入文档ID: ").strip()
            obj_type = input("对象类型 (默认docx): ").strip() or "docx"

            try:
                result = client.delete_document(
                    document_id=document_id, obj_type=obj_type
                )

                if result.get("code") == 0:
                    print(f"✅ 成功删除文档: {document_id}")
                else:
                    print(f"❌ 失败: {result.get('msg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "7":
            # 查看日志
            log_dir = os.path.expanduser("~/.feishu_sdk")
            log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".log")])

            print(f"📁 日志目录: {log_dir}")
            print(f"📊 日志文件数量: {len(log_files)}")
            print()

            for log_file in log_files[-5:]:  # 显示最近5个
                print(f"  - {log_file}")

            if log_files:
                latest_log = log_files[-1]
                full_path = os.path.join(log_dir, latest_log)

                print()
                print(f"📄 最新日志: {full_path}")
                print("-" * 80)

                with open(full_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[-50:]:  # 显示最后50行
                        print(line.rstrip())

                print("-" * 80)

        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生异常: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
