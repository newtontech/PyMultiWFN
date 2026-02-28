# 飞书文档集成 - 快速开始指南

> 创建时间: 2026-01-31 14:15
> SDK版本: 1.0.0
> Python版本: 3.11+

---

## 🚀 快速开始

### 步骤1: 获取飞书应用凭证

1. **访问飞书开放平台**
   - 网址: https://open.feishu.cn/app

2. **创建应用**
   - 点击"创建应用"
   - 选择"文档"类型
   - 填写应用名称和描述

3. **获取凭证**
   - 在应用详情页查看
   - 复制 `App ID` 和 `App Secret`
   - 在企业设置中查看 `Tenant ID`

4. **配置权限**
   - 在应用权限中添加：
     - `docx:document` (读取文档)
     - `docx:document:write` (写入文档)

---

## 📊 飞书文档SDK功能

### 核心功能

1. **查看文档列表**
   - 列出云空间中的所有文档
   - 支持分页

2. **读取文档内容**
   - 读取文档的完整内容
   - 支持多种文档类型

3. **写入文档内容**
   - 创建新文档
   - 更新现有文档

4. **搜索文档**
   - 关键词搜索
   - 全文搜索

5. **删除文档**
   - 安全删除文档

---

## 💻 使用方式

### 方式1: 交互式菜单（推荐）

```bash
cd ~/software/PyMultiWFN
source .venv/bin/activate
python3 feishu_doc_sdk.py
```

### 方式2: Python代码导入

```python
from feishu_doc_sdk import FeishuDocClient

# 创建客户端
client = FeishuDocClient(
    app_id="cli_xxxxxxxxxxxxxxx",
    app_secret="xxxxxxxxxxxxxxx",
    tenant_id="xxxxxxxxxxxxxxxx"
)

# 查看文档列表
result = client.list_documents(space_id="your_space_id")
print(result)
```

### 方式3: 作为库集成

```python
# 在你的项目中导入
from feishu_doc_sdk import FeishuDocClient, base64_encode_file

# 使用SDK
client = FeishuDocClient(app_id, app_secret, tenant_id)

# 创建文档
content = base64_encode_file("your_file.docx")
client.create_document(space_id="your_space_id", title="My Document", content=content)
```

---

## 📝 API参考

### 查看文档列表

```python
client.list_documents(
    space_id="space_id",
    page_size=50,
    page_token=""
)
```

**参数**:
- `space_id` (必填): 云空间ID
- `page_size` (可选): 每页数量，默认50
- `page_token` (可选): 分页token

**返回**:
```json
{
  "code": 0,
  "data": {
    "items": [...],
    "page_token": "",
    "has_more": false
  }
}
```

### 读取文档内容

```python
client.get_document(
    document_id="doc_id",
    obj_type="docx",
    obj_basic=True
)
```

**参数**:
- `document_id` (必填): 文档ID
- `obj_type` (必填): 对象类型 (docx, mindnote, wiki等)
- `obj_basic` (可选): 是否返回基本信息，默认true

**返回**:
```json
{
  "code": 0,
  "data": {
    "content": "document content...",
    "title": "Document Title"
  }
}
```

### 创建文档

```python
client.create_document(
    space_id="space_id",
    title="Document Title",
    content="base64_content",
    folder_token=""
)
```

**参数**:
- `space_id` (必填): 云空间ID
- `title` (必填): 文档标题
- `content` (必填): Base64编码的文件内容
- `folder_token` (可选): 文件夹token

**返回**:
```json
{
  "code": 0,
  "data": {
    "document_id": "doc_id",
    "revision_id": "rev_id"
  }
}
```

### 更新文档

```python
client.update_document(
    document_id="doc_id",
    obj_type="docx",
    title="New Title",
    content="base64_content"
)
```

**参数**:
- `document_id` (必填): 文档ID
- `obj_type` (必填): 对象类型
- `title` (可选): 新标题
- `content` (可选): Base64编码的新内容

**返回**:
```json
{
  "code": 0,
  "data": {
    "revision_id": "rev_id"
  }
}
```

### 删除文档

```python
client.delete_document(
    document_id="doc_id",
    obj_type="docx"
)
```

**参数**:
- `document_id` (必填): 文档ID
- `obj_type` (必填): 对象类型

**返回**:
```json
{
  "code": 0
  "data": {}
}
```

---

## 🔧 高级功能

### 日志记录

SDK会自动记录所有API调用到 `~/.feishu_sdk/` 目录：
- 日志文件格式: `feishu_YYYYMMDD.log`
- 日志内容: 时间戳、日志级别、消息

### 查看日志

1. 运行SDK脚本
2. 选择选项7: 查看日志
3. 查看最近5个日志文件
4. 查看每个日志文件的最后50行

### 错误处理

SDK会自动处理错误：
- 网络错误: 自动重试（可配置）
- API错误: 显示详细错误信息
- 认证错误: 提示检查凭证

---

## 📋 常见问题

### Q1: 如何获取Space ID?

**A**: 
1. 在飞书文档中打开云空间
2. 查看URL，例如: `https://feishu.cn/drive/folder/xxxxxx`
3. URL中的`xxxxxx`就是Space ID

### Q2: 如何获取Document ID?

**A**: 
1. 运行SDK的"查看文档列表"功能
2. 从返回的文档列表中查看`document_id`
3. 复制该ID用于其他操作

### Q3: 如何支持中文文件名?

**A**: 
1. 在创建或更新文档时，文件名使用UTF-8编码
2. Base64编码不受语言影响
3. SDK会自动处理编码

### Q4: API调用频率限制?

**A**: 
- 飞书API有频率限制
- 建议每次操作间隔至少1秒
- 批量操作建议使用异步队列

### Q5: 如何处理大文件?

**A**: 
1. 对于大文件(>10MB)，建议分块上传
2. 或者使用飞书的批量导入功能
3. SDK会自动处理Base64编码

---

## 🎯 使用示例

### 示例1: 创建PyMultiWFN文档

```python
from feishu_doc_sdk import FeishuDocClient, base64_encode_file

# 创建客户端
client = FeishuDocClient(
    app_id="your_app_id",
    app_secret="your_app_secret",
    tenant_id="your_tenant_id"
)

# 创建文档
content = base64_encode_file("README.md")
result = client.create_document(
    space_id="your_space_id",
    title="PyMultiWFN README",
    content=content
)

print(f"文档ID: {result['data']['document_id']}")
```

### 示例2: 更新现有文档

```python
from feishu_doc_sdk import FeishuDocClient, base64_encode_file

# 创建客户端
client = FeishuDocClient(
    app_id="your_app_id",
    app_secret="your_app_secret",
    tenant_id="your_tenant_id"
)

# 更新文档
content = base64_encode_file("updated_README.md")
result = client.update_document(
    document_id="doc_id",
    obj_type="docx",
    title="Updated Title",
    content=content
)

print(f"版本ID: {result['data']['revision_id']}")
```

### 示例3: 搜索特定文档

```python
# 搜索文档
result = client.search_documents(
    space_id="your_space_id",
    query="PyMultiWFN"
    page_size=50
)

# 显示结果
for item in result['data']['items']:
    print(f"{item['document_id']} - {item['title']}")
```

---

## 📚 更多资源

- [飞书开放平台文档](https://open.feishu.cn/document/)
- [飞书文档API参考](https://open.feishu.cn/docx/docs/)
- [飞书SDK GitHub](https://github.com/larksuite/oapi-sdks)

---

## 🎓 学习路径

1. **了解飞书文档API** (30分钟)
   - 阅读官方文档
   - 了解API结构和认证流程

2. **配置开发环境** (15分钟)
   - 获取应用凭证
   - 配置SDK
   - 运行示例代码

3. **实践基本功能** (1小时)
   - 查看文档列表
   - 读取文档内容
   - 创建新文档

4. **进阶功能** (2小时)
   - 搜索功能
   - 更新和删除
   - 错误处理

---

**SDK版本**: 1.0.0  
**最后更新**: 2026-01-31  
**维护者**: PyMultiWFN Team
