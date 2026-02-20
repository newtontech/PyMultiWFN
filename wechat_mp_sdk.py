#!/usr/bin/env python3
"""
微信公众号集成 - Python SDK
支持图文消息发送、素材管理、粉丝互动
"""

import os
import sys
import json
import time
import hashlib
import requests
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from urllib.parse import urlencode


class WeChatMediaPlatform:
    """微信公众号平台客户端"""

    def __init__(self, app_id: str, app_secret: str):
        """
        初始化微信公众号客户端

        Args:
            app_id: 微信公众号AppID
            app_secret: 微信公众号AppSecret
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = None
        self.token_expires = None
        self.base_url = "https://api.weixin.qq.com/cgi-bin"

        # 日志
        self.log_dir = os.path.expanduser("~/.wechat_mp")
        os.makedirs(self.log_dir, exist_ok=True)

    def _log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = os.path.join(
            self.log_dir, f"wechat_{datetime.now().strftime('%Y%m%d')}.log"
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")

        print(f"[{level}] {message}")

    def _get_access_token(self) -> str:
        """获取access_token"""
        if self.access_token and time.time() < self.token_expires:
            return self.access_token

        url = f"{self.base_url}/token"

        params = {
            "grant_type": "client_credential",
            "appid": self.app_id,
            "secret": self.app_secret,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            result = response.json()

            if result.get("errcode") != 0:
                self._log(f"获取token失败: {result}", "ERROR")
                raise Exception(f"获取token失败: {result.get('errmsg', '未知错误')}")

            self.access_token = result["access_token"]
            self.token_expires = time.time() + 7200 - 300  # 2小时-5分钟

            self._log(f"成功获取微信access_token", "SUCCESS")
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
        use_json: bool = True,
    ) -> Any:
        """发送API请求"""

        access_token = self._get_access_token()

        url = f"{self.base_url}{endpoint}"

        if use_json:
            headers = {"Content-Type": "application/json; charset=utf-8"}
            if params:
                params["access_token"] = access_token
        else:
            headers = {}
            if params:
                params["access_token"] = access_token

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == "POST":
                if files:
                    response = requests.post(
                        url, headers=headers, data=data, files=files, timeout=30
                    )
                else:
                    if use_json:
                        response = requests.post(
                            url, headers=headers, json=data, params=params, timeout=30
                        )
                    else:
                        response = requests.post(
                            url, headers=headers, data=data, params=params, timeout=30
                        )
            elif method == "UPLOAD":
                response = requests.post(url, headers=headers, files=files, timeout=30)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()
            return response.json()

        except Exception as e:
            self._log(f"API请求异常: {method} {endpoint} - {str(e)}", "ERROR")
            raise

    def get_material_count(self, material_type: str = "image") -> Dict[str, Any]:
        """
        获取素材总数

        Args:
            material_type: 素材类型

        Returns:
            素材总数响应
        """
        self._log(f"获取素材总数 - type: {material_type}")

        endpoint = f"/material/get_materialcount"

        data = {"type": material_type}

        return self._make_request("POST", endpoint, data=data)

    def batch_get_material(
        self, material_type: str = "image", offset: int = 0, count: int = 20
    ) -> Dict[str, Any]:
        """
        批量获取素材列表

        Args:
            material_type: 素材类型
            offset: 偏移量
            count: 返回数量

        Returns:
            素材列表响应
        """
        self._log(
            f"批量获取素材 - type: {material_type}, offset: {offset}, count: {count}"
        )

        endpoint = f"/material/batchget_material"

        data = {"type": material_type, "offset": offset, "count": count}

        return self._make_request("POST", endpoint, data=data)

    def add_news(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        新增永久素材

        Args:
            articles: 图文消息列表

        Returns:
            上传响应
        """
        self._log(f"新增永久素材 - 文章数: {len(articles)}")

        endpoint = f"/material/add_news"

        data = {"articles": articles}

        return self._make_request("POST", endpoint, data=data)

    def upload_image(self, file_path: str) -> Dict[str, Any]:
        """
        上传图片

        Args:
            file_path: 图片路径

        Returns:
            上传响应
        """
        self._log(f"上传图片 - file: {file_path}")

        endpoint = f"/material/add_material"

        # 打开文件
        with open(file_path, "rb") as f:
            files = {"media": f}

            data = {"type": "image"}

            # 上传
            return self._make_request("UPLOAD", endpoint, data=data, files=files)

    def upload_other(self, file_path: str, media_type: str = "video") -> Dict[str, Any]:
        """
        上传其他素材（视频、语音等）

        Args:
            file_path: 文件路径
            media_type: 媒体类型

        Returns:
            上传响应
        """
        self._log(f"上传其他素材 - file: {file_path}, type: {media_type}")

        endpoint = f"/material/add_material"

        # 打开文件
        with open(file_path, "rb") as f:
            files = {"media": f}

            data = {"type": media_type, "description": file_path}

            # 上传
            return self._make_request("UPLOAD", endpoint, data=data, files=files)

    def upload_news_material(
        self,
        file_path: str,
        title: str,
        author: str,
        digest: str,
        content_source_url: str,
        thumb_media_id: str,
    ) -> Dict[str, Any]:
        """
        新增其他类型永久素材

        Args:
            file_path: 文件路径
            title: 标题
            author: 作者
            digest: 文章内容
            content_source_url: 内容来源URL
            thumb_media_id: 缩略图media_id

        Returns:
            上传响应
        """
        self._log(f"新增永久图文素材 - title: {title}")

        endpoint = f"/material/add_news"

        # 打开文件
        with open(file_path, "rb") as f:
            files = {"media": f}

            data = {
                "title": title,
                "author": author,
                "digest": digest,
                "content_source_url": content_source_url,
                "thumb_media_id": thumb_media_id,
            }

            # 上传
            return self._make_request("UPLOAD", endpoint, data=data, files=files)

    def upload_img(self, file_path: str) -> Dict[str, Any]:
        """
        上传图片

        Args:
            file_path: 图片路径

        Returns:
            上传响应（包含url）
        """
        self._log(f"上传图片 - file: {file_path}")

        endpoint = f"/cgi-bin/media/uploadimg"

        # 打开文件
        with open(file_path, "rb") as f:
            files = {"media": f}

            data = {}

            # 上传
            return self._make_request(
                "UPLOAD", endpoint, data=data, files=files, use_json=False
            )

    def get_follower_list(self, next_openid: str = "") -> Dict[str, Any]:
        """
        获取粉丝列表

        Args:
            next_openid: 下一个拉取的OpenID

        Returns:
            粉丝列表响应
        """
        self._log(f"获取粉丝列表 - next_openid: {next_openid}")

        endpoint = f"/user/get"

        data = {"next_openid": next_openid}

        return self._make_request("POST", endpoint, data=data)

    def get_follower_info(self, openid: str) -> Dict[str, Any]:
        """
        获取用户基本信息

        Args:
            openid: 用户OpenID

        Returns:
            用户信息响应
        """
        self._log(f"获取用户信息 - openid: {openid}")

        endpoint = f"/user/info"

        data = {"openid": openid, "lang": "zh_CN"}

        return self._make_request("POST", endpoint, data=data)


def create_article(
    title: str,
    author: str,
    digest: str,
    content: str,
    content_source_url: str,
    thumb_media_id: str,
) -> Dict[str, Any]:
    """
    创建图文消息文章

    Args:
        title: 标题
        author: 作者
        digest: 文章内容摘要
        content: 文章内容（HTML）
        content_source_url: 内容来源URL
        thumb_media_id: 缩略图media_id

    Returns:
        文章字典
    """
    return {
        "title": title,
        "author": author,
        "digest": digest,
        "content": content,
        "content_source_url": content_source_url,
        "thumb_media_id": thumb_media_id,
    }


def interactive_menu():
    """交互式菜单"""

    print("=" * 80)
    print("  微信公众号集成 - Python SDK")
    print("=" * 80)
    print()

    print("请选择功能:")
    print()
    print("  1. 获取素材总数")
    print("  2. 批量获取素材列表")
    print("  3. 新增永久素材")
    print("  4. 上传图片")
    print("  5. 上传其他素材（视频、语音）")
    print("  6. 获取粉丝列表")
    print("  7. 获取用户信息")
    print("  8. 上传缩略图")
    print("  9. 查看日志")
    print("  0. 退出")
    print()

    return input("输入选项 (0-9): ")


def main():
    """主函数"""

    print()
    print("=" * 80)
    print("  微信公众号集成")
    print("=" * 80)
    print()
    print("功能:")
    print("  ✅ 获取素材总数")
    print("  ✅ 批量获取素材列表")
    print("  ✅ 新增永久素材")
    print("  ✅ 上传图片")
    print("  ✅ 上传其他素材")
    print("  ✅ 获取粉丝列表")
    print("  ✅ 获取用户信息")
    print("  ✅ 上传缩略图")
    print()
    print("=" * 80)
    print()

    # 配置信息
    config_file = os.path.expanduser("~/.wechat_mp/config.json")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)

    # 检查配置
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        print(f"📄 找到配置文件: {config_file}")
        print(f"  AppID: {config.get('app_id', '未设置')}")
        print(f"  AppSecret: {'*' * len(config.get('app_secret', ''))}")
        print()

        use_existing = input("使用现有配置? (y/n): ").strip().lower()

        if use_existing != "y":
            config = {}
    else:
        config = {}

    if not config.get("app_id") or not config.get("app_secret"):
        print("请配置微信公众号凭证:")
        print()
        config["app_id"] = input("AppID: ").strip()
        config["app_secret"] = input("AppSecret: ").strip()
        print()

        # 保存配置
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✅ 配置已保存到: {config_file}")
        print()

    # 创建客户端
    try:
        client = WeChatMediaPlatform(
            app_id=config["app_id"], app_secret=config["app_secret"]
        )

        print("✅ 微信公众号客户端创建成功")
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
            # 获取素材总数
            material_type = input("素材类型 (默认image): ").strip() or "image"

            try:
                result = client.get_material_count(material_type=material_type)

                if result.get("errcode") == 0:
                    count = result.get("item_count", 0)
                    print(f"✅ 成功获取素材总数: {count}")
                else:
                    print(f"❌ 失败: {result.get('errmsg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "2":
            # 批量获取素材列表
            material_type = input("素材类型 (默认image): ").strip() or "image"
            offset = input("偏移量 (默认0): ").strip() or "0"
            count = input("返回数量 (默认20): ").strip() or "20"

            try:
                result = client.batch_get_material(
                    material_type=material_type, offset=int(offset), count=int(count)
                )

                if result.get("errcode") == 0:
                    items = result.get("item", [])
                    print(f"✅ 成功获取素材列表: {len(items)} 个")

                    for i, item in enumerate(items[:10], 1):
                        media_id = item.get("media_id", "N/A")
                        name = item.get("name", "N/A")
                        url = item.get("url", "N/A")
                        print(f"  {i}. {media_id} - {name} - {url}")

                    if len(items) > 10:
                        print(f"  ... 还有 {len(items) - 10} 个素材")

                else:
                    print(f"❌ 失败: {result.get('errmsg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "3":
            # 新增永久素材
            print("新增永久图文素材:")
            print()

            # 读取文件列表
            file_paths = []
            while True:
                file_path = input(f"文件路径 {len(file_paths)+1} (回车结束): ").strip()
                if not file_path:
                    break
                file_paths.append(file_path)

            if not file_paths:
                print("⚠️  未输入文件路径")
                continue

            # 创建文章列表
            articles = []
            for file_path in file_paths:
                title = input(f"文件 {file_path} 的标题: ").strip()
                author = input("作者: ").strip()
                digest = input("摘要: ").strip()
                thumb_media_id = input("缩略图media_id (可选): ").strip()
                content_source_url = input("内容来源URL (可选): ").strip()

                # 读取文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 创建文章
                article = create_article(
                    title=title,
                    author=author,
                    digest=digest,
                    content=content,
                    content_source_url=content_source_url,
                    thumb_media_id=thumb_media_id or "",
                )

                articles.append(article)

            # 上传
            try:
                result = client.add_news(articles)

                if result.get("errcode") == 0:
                    media_id = result.get("media_id", "N/A")
                    print(f"✅ 成功新增永久素材: {media_id}")
                    print(f"  文章数: {len(articles)}")
                else:
                    print(f"❌ 失败: {result.get('errmsg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "4":
            # 上传图片
            file_path = input("图片文件路径: ").strip()

            try:
                result = client.upload_image(file_path)

                if result.get("errcode") == 0:
                    media_id = result.get("media_id", "N/A")
                    url = result.get("url", "N/A")
                    print(f"✅ 成功上传图片")
                    print(f"  media_id: {media_id}")
                    print(f"  url: {url}")
                else:
                    print(f"❌ 失败: {result.get('errmsg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "5":
            # 上传其他素材
            file_path = input("文件路径: ").strip()
            media_type = input("媒体类型 (默认video): ").strip() or "video"

            try:
                result = client.upload_other(file_path, media_type=media_type)

                if result.get("errcode") == 0:
                    media_id = result.get("media_id", "N/A")
                    print(f"✅ 成功上传{media_type}")
                    print(f"  media_id: {media_id}")
                else:
                    print(f"❌ 失败: {result.get('errmsg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "6":
            # 获取粉丝列表
            next_openid = input("下一个OpenID (可选，回车跳过): ").strip() or ""

            try:
                result = client.get_follower_list(next_openid=next_openid)

                if result.get("errcode") == 0:
                    data = result.get("data", {})
                    total = data.get("total", 0)
                    count = data.get("count", 0)
                    openids = data.get("data", {}).get("openid", [])

                    print(f"✅ 成功获取粉丝列表")
                    print(f"  总粉丝数: {total}")
                    print(f"  本次拉取: {count}")
                    print(f"  OpenIDs: {len(openids)}")

                    for i, openid in enumerate(openids[:10], 1):
                        print(f"  {i}. {openid}")

                    if len(openids) > 10:
                        print(f"  ... 还有 {len(openids) - 10} 个粉丝")

                    if data.get("next_openid"):
                        print(f"  next_openid: {data.get('next_openid')}")

                else:
                    print(f"❌ 失败: {result.get('errmsg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "7":
            # 获取用户信息
            openid = input("用户OpenID: ").strip()

            try:
                result = client.get_follower_info(openid=openid)

                if result.get("errcode") == 0:
                    data = result.get("data", {})
                    subscribe = data.get("subscribe", 0)
                    openid = data.get("openid", "N/A")
                    nickname = data.get("nickname", "N/A")
                    sex = data.get("sex", 0)
                    language = data.get("language", "N/A")
                    city = data.get("city", "N/A")
                    province = data.get("province", "N/A")
                    country = data.get("country", "N/A")
                    headimgurl = data.get("headimgurl", "N/A")
                    subscribe_time = data.get("subscribe_time", "N/A")

                    print(f"✅ 成功获取用户信息")
                    print(f"  OpenID: {openid}")
                    print(f"  昵称: {nickname}")
                    print(f"  关注状态: {subscribe}")
                    print(f"  语言: {language}")
                    print(f"  城市: {city}")
                    print(f"  头像: {headimgurl}")

                else:
                    print(f"❌ 失败: {result.get('errmsg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "8":
            # 上传缩略图
            file_path = input("图片文件路径: ").strip()

            try:
                result = client.upload_img(file_path)

                if result.get("errcode") == 0:
                    url = result.get("url", "N/A")
                    print(f"✅ 成功上传缩略图")
                    print(f"  url: {url}")
                else:
                    print(f"❌ 失败: {result.get('errmsg', '未知错误')}")

            except Exception as e:
                print(f"❌ 异常: {str(e)}")

        elif choice == "9":
            # 查看日志
            log_dir = os.path.expanduser("~/.wechat_mp")
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
