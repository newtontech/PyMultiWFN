# 微信公众号集成 - 快速开始指南

> 创建时间: 2026-01-31 14:45
> SDK版本: 1.0.0
> Python版本: 3.11+

---

## 🚀 快速开始

### 步骤1: 获取微信公众号凭证

1. **访问微信公众平台**
   - 网址: https://mp.weixin.qq.com/
   - 登录公众号后台

2. **开发配置**
   - 点击"开发" > "基本配置"
   - 获取 AppID 和 AppSecret

3. **服务器配置**
   - 填写服务器地址
   - 填写令牌 (Token)
   - 选择消息加解密方式

---

## 📊 微信公众号API功能

### 核心功能

1. **素材管理**
   - 获取素材总数
   - 批量获取素材列表
   - 新增永久素材
   - 上传图片
   - 上传其他素材（视频、语音）
   - 上传缩略图

2. **粉丝管理**
   - 获取粉丝列表
   - 获取用户基本信息
   - 获取用户OpenID

3. **消息推送**
   - 群发消息（图文、图片、文本等）
   - 预览接口
   - 消息状态查询

---

## 💻 使用方式

### 方式1: 交互式菜单（推荐）

```bash
cd ~/software/PyMultiWFN
source .venv/bin/activate
python3 wechat_mp_sdk.py
```

### 方式2: Python代码导入

```python
from wechat_mp_sdk import WeChatMediaPlatform, create_article

# 创建客户端
client = WeChatMediaPlatform(
    app_id="your_app_id",
    app_secret="your_app_secret"
)

# 获取素材总数
result = client.get_material_count(material_type="image")
print(result)
```

### 方式3: 作为库集成

```python
# 在你的项目中导入
from wechat_mp_sdk import WeChatMediaPlatform, create_article

# 使用SDK
client = WeChatMediaPlatform(app_id, app_secret)

# 上传图片
result = client.upload_image("your_image.jpg")
media_id = result['media_id']

# 创建图文文章
articles = [create_article(...)]
result = client.add_news(articles)
```

---

## 📝 API参考

### 获取素材总数

```python
client.get_material_count(
    material_type="image"
)
```

**参数**:
- `material_type` (必填): 素材类型
  - `image`: 图片
  - `voice`: 语音
  - `video`: 视频
  - `thumb`: 缩略图

**返回**:
```json
{
  "errcode": 0,
  "errmsg": "ok",
  "item_count": 12345
}
```

### 批量获取素材列表

```python
client.batch_get_material(
    material_type="image",
    offset=0,
    count=20
)
```

**参数**:
- `material_type` (必填): 素材类型
- `offset` (可选): 从第几个素材开始
- `count` (可选): 返回多少个素材

**返回**:
```json
{
  "errcode": 0,
  "errmsg": "ok",
  "item_count": 12345,
  "item": [...],
  "total_count": 12345
}
```

### 新增永久素材

```python
client.add_news(
    articles=[
        {
            "title": "文章标题",
            "author": "作者",
            "digest": "摘要",
            "content": "HTML内容",
            "content_source_url": "文章来源",
            "thumb_media_id": "缩略图media_id"
        }
    ]
)
```

**参数**:
- `articles` (必填): 图文消息文章列表

**文章字段**:
- `title`: 标题
- `author`: 作者
- `digest`: 摘要
- `content`: HTML内容
- `content_source_url`: 文章来源URL
- `thumb_media_id`: 缩略图media_id

**返回**:
```json
{
  "errcode": 0,
  "errmsg": "ok",
  "media_id": "media_id",
  "created_at": 1234567890
}
```

### 上传图片

```python
client.upload_image(
    file_path="/path/to/image.jpg"
)
```

**参数**:
- `file_path` (必填): 图片文件路径

**返回**:
```json
{
  "errcode": 0,
  "errmsg": "ok",
  "type": "image",
  "media_id": "media_id",
  "created_at": 1234567890
}
```

### 上传其他素材

```python
client.upload_other(
    file_path="/path/to/video.mp4",
    media_type="video"
)
```

**参数**:
- `file_path` (必填): 文件路径
- `media_type` (必填): 媒体类型
  - `video`: 视频
  - `voice`: 语音

**返回**:
```json
{
  "errcode": 0,
  "errmsg": "ok",
  "type": "video",
  "media_id": "media_id",
  "created_at": 1234567890
}
```

### 获取粉丝列表

```python
client.get_follower_list(
    next_openid=""
)
```

**参数**:
- `next_openid` (可选): 下一个拉取的OpenID

**返回**:
```json
{
  "errcode": 0,
  "errmsg": "ok",
  "total": 100000,
  "count": 10000,
  "data": {
    "openid": ["openid1", "openid2", ...],
    "next_openid": "openid_next"
  }
}
```

### 获取用户信息

```python
client.get_follower_info(
    openid="user_openid"
)
```

**参数**:
- `openid` (必填): 用户OpenID

**返回**:
```json
{
  "errcode": 0,
  "errmsg": "ok",
  "subscribe": 1,
  "openid": "user_openid",
  "nickname": "昵称",
  "sex": 1,
  "language": "zh_CN",
  "city": "上海",
  "province": "上海",
  "country": "中国",
  "headimgurl": "https://..."
}
```

### 上传缩略图

```python
client.upload_img(
    file_path="/path/to/thumb.jpg"
)
```

**参数**:
- `file_path` (必填): 图片文件路径

**返回**:
```json
{
  "errcode": 0,
  "errmsg": "ok",
  "url": "https://..."
}
```

---

## 🔧 高级功能

### 日志记录

SDK会自动记录所有API调用到 `~/.wechat_mp/` 目录：
- 日志文件格式: `wechat_YYYYMMDD.log`
- 日志内容: 时间戳、日志级别、消息

### 查看日志

1. 运行SDK脚本
2. 选择选项9: 查看日志
3. 查看最近5个日志文件
4. 查看每个日志文件的最后50行

### 错误处理

SDK会自动处理错误：
- 网络错误: 自动重试（可配置）
- API错误: 显示详细错误信息
- 认证错误: 提示检查凭证

---

## 📋 常见问题

### Q1: 如何获取AppID和AppSecret?

**A**: 
1. 访问: https://mp.weixin.qq.com/
2. 登录公众号后台
3. 点击"开发" > "基本配置"
4. 在"开发者ID(AppID)"和"开发者密码"中查看

### Q2: 如何获取粉丝OpenID?

**A**: 
1. 运行SDK的"获取粉丝列表"功能
2. 从返回的`data.openid`数组中查看
3. 复制该OpenID用于其他操作

### Q3: 如何支持中文文件名?

**A**: 
1. 上传文件时，SDK会自动处理编码
2. 微信API支持UTF-8编码
3. 文件名在服务器上会自动处理

### Q4: API调用频率限制?

**A**: 
- 微信API有频率限制
- 建议每次操作间隔至少1秒
- 批量操作建议使用异步队列

### Q5: 如何处理大文件?

**A**: 
1. 图片: 支持上传，建议<2M
2. 视频: 支持上传，建议<10M
3. 语音: 支持上传，建议<2M
4. 超大文件: 建议使用分片上传或CDN

---

## 🎯 使用示例

### 示例1: 上传PyMultiWFN封面图

```python
from wechat_mp_sdk import WeChatMediaPlatform

# 创建客户端
client = WeChatMediaPlatform(
    app_id="your_app_id",
    app_secret="your_app_secret"
)

# 上传封面图
result = client.upload_image("pymultiwfn_cover.jpg")
media_id = result['media_id']

print(f"封面图media_id: {media_id}")
```

### 示例2: 创建PyMultiWFN图文文章

```python
from wechat_mp_sdk import WeChatMediaPlatform, create_article

# 创建客户端
client = WeChatMediaPlatform(
    app_id="your_app_id",
    app_secret="your_app_secret"
)

# 创建文章
article = create_article(
    title="PyMultiWFN - 量子化学分析工具",
    author="PyMultiWFN Team",
    digest="PyMultiWFN是一个强大的Python量子化学分析工具",
    content="<html>...</html>",
    content_source_url="https://github.com/chemoinfolabs/pymultiwfn",
    thumb_media_id="cover_media_id"
)

# 新增永久素材
result = client.add_news([article])

print(f"文章media_id: {result['media_id']}")
```

### 示例3: 批量获取素材

```python
# 获取所有图片素材
offset = 0
page_size = 20

while True:
    result = client.batch_get_material(
        material_type="image",
        offset=offset,
        count=page_size
    )
    
    items = result.get('item', [])
    
    # 处理素材
    for item in items:
        media_id = item.get('media_id')
        name = item.get('name')
        url = item.get('url')
        update_time = item.get('update_time')
        
        print(f"{media_id} - {name} - {url}")
    
    # 检查是否还有更多
    if len(items) < page_size:
        break
    
    offset += page_size
```

---

## 📚 更多资源

- [微信公众平台文档](https://developers.weixin.qq.com/doc/offiaccount/)
- [微信公众号API参考](https://developers.weixin.qq.com/doc/offiaccount/Getting_Started/Overview)
- [微信公众号开发指南](https://developers.weixin.qq.com/doc/offiaccount/Getting_Started/Overview)

---

## 🎓 学习路径

1. **了解微信公众号API** (30分钟)
   - 阅读官方文档
   - 了解API结构和认证流程

2. **配置开发环境** (15分钟)
   - 获取AppID和AppSecret
   - 配置SDK
   - 运行示例代码

3. **实践基本功能** (1小时)
   - 获取素材总数
   - 批量获取素材
   - 上传图片
   - 获取粉丝列表

4. **进阶功能** (2小时)
   - 新增永久素材
   - 上传其他素材
   - 创建图文文章
   - 获取用户信息

---

**SDK版本**: 1.0.0  
**最后更新**: 2026-01-31  
**维护者**: PyMultiWFN Team
