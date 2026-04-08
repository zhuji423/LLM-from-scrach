import tiktoken

# 初始化 GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 测试字符串
string_id = "AARgumauri我爱你"
encode_ids = tokenizer.encode(string_id)

print("=" * 60)
print("1. 完整编码结果")
print("=" * 60)
print(f"原始字符串: {string_id}")
print(f"Token IDs: {encode_ids}")
print(f"Token 数量: {len(encode_ids)}")

print("\n" + "=" * 60)
print("2. 逐个 Token 解码（会出现乱码）")
print("=" * 60)
for idx, token_id in enumerate(encode_ids):
    decoded = tokenizer.decode([token_id])
    # 显示原始字节表示
    byte_repr = decoded.encode('utf-8', errors='replace')
    print(f"Token {idx}: ID={token_id:5d} | Decoded='{decoded}' | Bytes={byte_repr}")

print("\n" + "=" * 60)
print("3. 完整解码（正确显示）")
print("=" * 60)
decoded_full = tokenizer.decode(encode_ids)
print(f"完整解码: {decoded_full}")

print("\n" + "=" * 60)
print("4. 查看中文字符的字节拆分")
print("=" * 60)
chinese_char = "我"
print(f"字符: {chinese_char}")
print(f"UTF-8 字节: {chinese_char.encode('utf-8').hex()}")
print(f"Token IDs: {tokenizer.encode(chinese_char)}")

# 解释每个 token 对应的字节
for token_id in tokenizer.encode(chinese_char):
    decoded = tokenizer.decode([token_id])
    try:
        byte_val = decoded.encode('utf-8')
        print(f"  Token {token_id}: 字节 {byte_val.hex()}")
    except:
        print(f"  Token {token_id}: 无法解码为有效 UTF-8")
