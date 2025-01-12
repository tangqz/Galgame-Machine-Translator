import os
import json
import time
import re
import logging
import threading
import sys
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI
from tqdm import tqdm
from colorama import Fore, Style, init

# 强制将标准输出和标准错误设置为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 初始化 colorama
init(autoreset=True)

NEWLINE_PLACEHOLDER = "<__NEWLINE_PLACEHOLDER__>"
# 配置日志
log_filename = f"translation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8'
)
# 移除默认的控制台处理器
logging.getLogger('').handlers = []
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger('').addHandler(file_handler)

# 设置OpenAI API密钥和基础URL
DEEPSEEK_API_KEY = ""  # Deepseek API密钥
OPENAI_API_KEY = ""  # OpenAI API密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 创建两个客户端实例
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)  # 不设置base_url，使用默认的OpenAI端点

# 定义输入和输出文件夹
INPUT_FOLDER = r''
OUTPUT_FOLDER = r''

# 如果输出文件夹不存在，则创建
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 定义每批次对话数
CONV_PER_BATCH = 50
MAX_RETRIES = 10

# 定义一个锁用于线程安全地更新翻译结果
lock = threading.Lock()

def collect_all_roles(input_folder):
    """
    扫描所有输入文件，收集所有独特的角色名。
    """
    all_roles = set()
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.json')]

    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                for entry in data:
                    role = entry.get('name', '旁白')
                    all_roles.add(role)
        except Exception as e:
            logging.error(f"错误 - 文件: {file_path} | 无法读取或解析文件 - {str(e)}")

    return list(all_roles)

def translate_roles(all_roles, role_translation_file):
    """
    将所有独特的角色名发送给模型统一翻译，并保存翻译结果到文件。
    如果翻译文件已存在，加载现有翻译并补充新角色的翻译。
    """
    existing_translations = {}
    new_roles = set(all_roles)
    
    # 如果翻译文件存在，加载现有翻译
    if os.path.exists(role_translation_file):
        with open(role_translation_file, 'r', encoding='utf-8') as f:
            existing_translations = json.load(f)
            logging.info(f"加载现有的角色名翻译映射，共 {len(existing_translations)} 个角色。")
            
            # 找出需要新翻译的角色
            new_roles = set(all_roles) - set(existing_translations.keys())
            if new_roles:
                logging.info(f"发现 {len(new_roles)} 个新角色需要翻译。")
    
    # 如果有新角色需要翻译
    if new_roles:
        new_roles_list = list(new_roles)
        batches = [new_roles_list[i:i + CONV_PER_BATCH] for i in range(0, len(new_roles_list), CONV_PER_BATCH)]
        logging.info(f"开始翻译 {len(new_roles)} 个新角色名，分为 {len(batches)} 个批次。")

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {executor.submit(translate_role_batch, batch): batch for batch in batches}
            for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="翻译新角色名"):
                batch = future_to_batch[future]
                try:
                    translated_batch = future.result()
                    if translated_batch:
                        existing_translations.update(translated_batch)
                except Exception as e:
                    logging.error(f"新角色名翻译失败 - 批次: {batch} | 错误: {str(e)}")
                    raise

        # 保存更新后的翻译结果
        with open(role_translation_file, 'w', encoding='utf-8') as f:
            json.dump(existing_translations, f, ensure_ascii=False, indent=2)
        logging.info(f"角色名翻译更新完成，当前共有 {len(existing_translations)} 个角色翻译。")

    return existing_translations

def translate_role_batch(batch):
    """
    翻译一个批次的角色名。
    """
    system_prompt = "请将以下角色名从日文翻译成中文。保持名称的一致性和准确性。"
    # 将每个角色名中的换行符替换为占位符（虽然角色名一般不包含换行符，但为了保险起见）
    escaped_batch = [line.replace("\n", NEWLINE_PLACEHOLDER) for line in batch]
    user_content = "\n".join(escaped_batch)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3,
        stream=False
    )

    translated_text = response.choices[0].message.content.strip()
    # 将占位符替换回换行符
    translated_roles = [line.replace(NEWLINE_PLACEHOLDER, "\n") for line in translated_text.split('\n')]

    batch_translation = {}
    for original, translated in zip(batch, translated_roles):
        batch_translation[original] = translated

    return batch_translation

def translate_batch(file_name, batch_num, batch, system_prompt, role_translations, file_progress, translated_results):
    """
    翻译单个批次的对话。
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # 将每个对话行内的换行符替换为占位符
            escaped_batch = [line.replace("\n", NEWLINE_PLACEHOLDER) for line in batch]
            request_content = "\n".join(escaped_batch)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_content}
            ]

            # 记录请求信息
            logging.info(f"API请求 - 文件: {file_name} | 批次: {batch_num} | 尝试: {retry_count + 1}/{MAX_RETRIES}")
            logging.info(f"请求内容: {json.dumps(messages, ensure_ascii=False)}")

            # 根据重试次数选择API
            if retry_count >= 3:
                current_client = openai_client
                model = "gpt-4o-2024-11-20"
                temperature = 0.3
                logging.info(f"切换到OpenAI API - 文件: {file_name} | 批次: {batch_num}")
            else:
                current_client = deepseek_client
                model = "deepseek-chat"
                temperature = 1.2 if retry_count > 1 else 0.3

            # 发送API请求
            response = current_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False
            )

            # 记录完整的API响应
            response_data = {
                'id': response.id,
                'created': response.created,
                'model': response.model,
                'content': response.choices[0].message.content,
                'finish_reason': response.choices[0].finish_reason,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            logging.info(f"API响应 - 文件: {file_name} | 批次: {batch_num}")
            logging.info(f"响应数据: {json.dumps(response_data, ensure_ascii=False, indent=2)}")

            api_response = response.choices[0].message.content.strip()

            # 将占位符替换回换行符
            translated_batch = [line.strip() for line in api_response.split('\n')]

            # 验证翻译结果
            if len(translated_batch) != len(batch):
                tqdm.write(Fore.YELLOW + f"警告 - {file_name} | 批次 {batch_num}: 翻译条数不一致（期望: {len(batch)}, 实际: {len(translated_batch)}），重试中...")
                logging.warning(f"翻译条数不一致 - 文件: {file_name} | 批次: {batch_num} | 期望: {len(batch)}, 实际: {len(translated_batch)}")
                retry_count += 1
                wait_time = min(2 * retry_count, 10)
                time.sleep(wait_time)
                continue

            # 角色名一致性验证
            roles_match = True
            for original_line, translated_line in zip(batch, translated_batch):
                # 使用更严格的正则表达式匹配，并允许多余空格
                original_match = re.match(r'(?s)^\[\s*(\d+)\s*\]\[\s*(.*?)\s*\]\[\s*(.*?)\s*\]$', original_line.strip())
                translated_match = re.match(r'^\[\s*(\d+)\s*\]\[\s*(.*?)\s*\]\[\s*(.*?)\s*\]$', translated_line.strip())

                if not (original_match and translated_match):
                    roles_match = False
                    logging.warning(f"格式不匹配 - 原文: {original_line}, 译文: {translated_line}")
                    break

                original_idx, original_role, _ = original_match.groups()
                translated_idx, translated_role, _ = translated_match.groups()

                # 清理并统一大小写
                original_role = original_role.strip()
                translated_role = translated_role.strip()

                # 验证编号一致
                if original_idx != translated_idx:
                    roles_match = False
                    logging.warning(f"编号不匹配 - 原文编号: {original_idx}, 译文编号: {translated_idx}")
                    break

                # 验证角色名翻译是否符合映射
                expected_role = role_translations.get(original_role, "旁白")
                if translated_role != expected_role:
                    roles_match = False
                    logging.warning(f"角色名不匹配 - 原文: {original_role}, 期望译文: {expected_role}, 实际译文: {translated_role}")
                    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\uFF66-\uFF9F]', translated_role):
                        logging.warning(f"译文角色名包含日文字符: {translated_role}")
                    break

            if roles_match:
                # 所有验证通过，添加翻译结果
                for i, translation in enumerate(translated_batch):
                    translation = translation.replace(NEWLINE_PLACEHOLDER, "\\n")
                    translated_results.append((batch_num * CONV_PER_BATCH + i, translation))
                file_progress.update(len(batch))
                logging.info(f"文件: {file_name} | 批次: {batch_num} 翻译成功")
                return  # 成功翻译，直接返回
            else:
                retry_count += 1
                wait_time = min(2 * retry_count, 10)
                tqdm.write(Fore.YELLOW + f"警告 - {file_name} | 批次 {batch_num}: 验证失败，重试中...")
                time.sleep(wait_time)
                continue

        except Exception as e:
            logging.error(f"API错误 - 文件: {file_name} | 批次: {batch_num} | 错误: {str(e)}")
            retry_count += 1
            wait_time = min(2 * retry_count, 10)
            tqdm.write(Fore.RED + f"错误 - {file_name} | 批次 {batch_num}: {str(e)[:100]}...")
            tqdm.write(Fore.YELLOW + f"重试 {retry_count}/{MAX_RETRIES} - {file_name} | 批次 {batch_num}，等待 {wait_time} 秒...")
            time.sleep(wait_time)

def translate_file(file_path, output_folder, batch_executor, role_translations):
    """
    处理单个文件的翻译工作。更新对translate_batch的调用。
    """
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, file_name)

    # 检查输出文件是否已存在
    if os.path.exists(output_file_path):
        tqdm.write(Fore.YELLOW + f"跳过 - {file_name} 已存在于输出文件夹中")
        logging.info(f"跳过 - 文件: {file_name} 已存在于输出文件夹中")
        return

    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        # 编码：转换为 [编号][角色名][对话内容] 格式
        encoded_lines = []
        for idx, entry in enumerate(data, 1):
            role = entry.get('name', '旁白')
            message = entry['message']
            line = f"[{idx}][{role}][{message}]"
            encoded_lines.append(line)

        # 构建角色名提示词（仅包含该批次中的角色）
        # 因为角色翻译已预先完成，因此不需要在每个请求中包含完整的映射

        # 将编码后的行分割成批次
        batches = [encoded_lines[i:i + CONV_PER_BATCH] for i in range(0, len(encoded_lines), CONV_PER_BATCH)]

        # 在文件处理开始时记录日志
        logging.info(f"文件: {file_name} | 开始翻译，共 {len(batches)} 个批次。")

        # 创建进度条，使用对话总数而不是批次数
        total_dialogues = len(encoded_lines)
        pbar = tqdm(total=total_dialogues, desc=f"翻译 {file_name}", leave=True, ncols=100)

        translated_results = []
        translation_failed = False  # 添加标志来追踪是否有批次翻译失败

        # 提交所有批次的翻译任务到批次线程池
        future_to_batch = {}
        for batch_num, batch in enumerate(batches, 1):
            # 提取当前批次中涉及的角色名
            batch_roles = set()
            for line in batch:
                match = re.match(r'\[(\d+)\]\[(.*?)\]\[(.*)\]', line)
                if match:
                    batch_roles.add(match.group(2))

            # 构建局部角色名映射
            local_role_mapping = {role: role_translations.get(role, "旁白") for role in batch_roles}

            # 构建角色名提示词
            role_prompt = "角色名对照表（请严格按照此表使用角色名）：\n"
            for original_role, translated_role in local_role_mapping.items():
                role_prompt += f"- {original_role}: {translated_role}\n"

            # 修改系统提示词，仅包含当前批次涉及的角色
            system_prompt = f"""请将以下视觉小说的对话文本从日文翻译成中文。请严格遵守以下规则：
1. 保持与输入的对话数量一致
2. 输出格式必须与输入相同：[编号][角色名][对话内容]
3. 角色名必须严格按照下方对照表使用中文名，禁止使用原日文名
4. 不要重复或合并对话
5. 确保编号一致且顺序正确
6. 请尤其注意，不要过多重复同一个词，使句子变得不自然，如ツッ……オオオオオオッッッ────翻译为呜──哦哦哦哦哦哦────即可。禁止输出过多个呜、哦等语气词，否则可能造成程序崩溃。
7. 如果违反规则，整批次将重新翻译

{role_prompt}"""

            future = batch_executor.submit(
                translate_batch,
                file_name,
                batch_num,
                batch,
                system_prompt,
                role_translations,
                pbar,
                translated_results
            )
            future_to_batch[future] = batch_num

        # 等待所有批次完成
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                future.result()
            except Exception as e:
                translation_failed = True  # 标记翻译失败
                tqdm.write(Fore.RED + f"错误 - {file_name} | 批次 {batch_num}: 达到最大重试次数，翻译失败 - {str(e)}")
                logging.error(f"错误 - 文件: {file_name} | 批次: {batch_num} | 翻译失败 - {str(e)}")
                break  # 一旦有批次失败就退出循环

        pbar.close()

        # 只有在所有批次都成功翻译的情况下才保存文件
        if not translation_failed:
            # 重新排序翻译结果
            translated_results_sorted = sorted(translated_results, key=lambda x: x[0])
            translated_lines = [line for idx, line in translated_results_sorted]

            # 验证是否所有对话都已翻译
            if len(translated_lines) != len(encoded_lines):
                raise Exception(f"翻译结果数量不匹配：期望 {len(encoded_lines)}，实际 {len(translated_lines)}")

            # 解码和保存文件的代码
            translated_data = []
            for line in translated_lines:
                match = re.match(r'\[(\d+)\]\[(.*?)\]\[(.*)\]', line)
                if match:
                    idx, role, message = match.groups()
                    entry = {"message": message}
                    if role != "旁白":
                        entry["name"] = role
                    translated_data.append(entry)
                else:
                    logging.warning(f"文件: {file_name} | 无法解析翻译后的行: {line}")

            # 保存翻译后的JSON文件
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(translated_data, outfile, ensure_ascii=False, indent=2)

            tqdm.write(Fore.GREEN + f"完成 - {file_name} 翻译完成")
            logging.info(f"完成 - 文件: {file_name} 翻译完成")
        else:
            tqdm.write(Fore.RED + f"失败 - {file_name} 翻译失败，跳过该文件")
            logging.error(f"失败 - 文件: {file_name} 翻译失败，跳过该文件")

    except Exception as e:
        tqdm.write(Fore.RED + f"错误 - {file_name}: {str(e)}")
        logging.error(f"错误 - 文件: {file_name} | 错误: {str(e)}")

def main():
    """
    主函数，处理所有文件的翻译。
    """
    # 获取所有JSON文件路径
    all_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]

    if not all_files:
        print("错误：输入文件夹中没有找到JSON文件。")
        return

    print(f"开始翻译 {len(all_files)} 个文件...")

    # 步骤1：收集所有角色名
    all_roles = collect_all_roles(INPUT_FOLDER)
    logging.info(f"收集到 {len(all_roles)} 个独特的角色名。")

    # 步骤2：翻译角色名
    role_translation_file = os.path.join(OUTPUT_FOLDER, "role_translations.json")
    role_translations = translate_roles(all_roles, role_translation_file)

    # 定义文件级别和批次级别的线程池
    max_file_workers = min(16, os.cpu_count() or 1)
    max_batch_workers = min(64, os.cpu_count() * 8 or 1)

    with ThreadPoolExecutor(max_workers=max_file_workers) as file_executor, \
         ThreadPoolExecutor(max_workers=max_batch_workers) as batch_executor:
        # 为每个文件提交一个翻译任务
        futures = {file_executor.submit(translate_file, file_path, OUTPUT_FOLDER, batch_executor, role_translations): file_path 
                  for file_path in all_files}
        
        for future in as_completed(futures):
            file_path = futures[future]
            file_name = os.path.basename(file_path)
            try:
                future.result()
            except Exception as e:
                tqdm.write(Fore.RED + f"错误 - {file_name}: 线程执行失败 - {str(e)}")
                logging.error(f"错误 - 文件: {file_name} | 线程执行失败 - {str(e)}")

    print("\n所有文件翻译完成。")
    logging.info("所有文件翻译完成。")

if __name__ == "__main__":
    main()