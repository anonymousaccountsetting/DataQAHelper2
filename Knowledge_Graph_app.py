import streamlit as st
import rdflib
import os
import glob
import zipfile
import io
import json
from collections import defaultdict
from rdflib import Graph, URIRef, BNode, Literal,Namespace
from rdflib.namespace import RDF,RDFS, OWL
from urllib.parse import quote

def load_config():
    """加载配置文件"""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            paths = config.get('kb_paths', {})
            return paths
    except FileNotFoundError:
        st.error("未找到配置文件 config.json！")
        return {}
    except json.JSONDecodeError:
        st.error("配置文件格式错误！")
        return {}

# 加载问题
def load_questions_from_ttl(ttl_file_path):
    g = rdflib.Graph()
    try:
        # 首先尝试读取文件内容
        with open(ttl_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查并修正常见的语法错误
        content = content.replace('"\n', '" .\n')  # 确保每个语句以 . 结尾
        content = content.replace('\r\n', '\n')    # 统一换行符
        
        # 修正缺少分号的问题
        lines = content.split('\n')
        fixed_lines = []
        in_subject_block = False
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                fixed_lines.append(line)
                continue
                
            if line.endswith(' a owl:Ontology ;'):
                in_subject_block = True
                fixed_lines.append(line)
            elif in_subject_block:
                if line.startswith(':'):
                    if i < len(lines) - 1 and lines[i + 1].startswith(':'):
                        if not line.endswith(';'):
                            line = line + ' ;'
                    elif not line.endswith('.'):
                        line = line + ' .'
                    if line.endswith(' .'):
                        in_subject_block = False
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # # 使用修正后的内容解析
        # g.parse(data=content, format="turtle")
        g.parse(
            data=content,
            format="turtle",
            publicID="http://example.org/ontology#"  # 添加基础 URI
        )
    except Exception as e:
        st.error(f"[解析错误] 无法解析 {ttl_file_path}: {str(e)}")
        return []

    # 更新后的 SPARQL 查询
    query = """
    PREFIX : <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?label ?tag ?optimDir ?overfitRisk
    WHERE {
      ?rule :hasQuestion ?qEntity .
      ?qEntity rdfs:label ?label .
      OPTIONAL { ?qEntity :hasTag ?tag. }
      OPTIONAL { ?qEntity :optimizationDirection ?optimDir. }
      OPTIONAL { ?qEntity :overfittingRisk ?overfitRisk. }
    }
    """

    question_list = []
    try:
        for row in g.query(query):
            q_label = str(row.label)
            q_tag = str(row.tag) if row.tag else None
            q_optim_dir = str(row.optimDir) if row.optimDir else None
            q_overfit = str(row.overfitRisk) if row.overfitRisk else None

            # 存成一个dict, 便于后续使用
            question_list.append({
                "text": q_label,
                "tag": q_tag,
                "optim_dir": q_optim_dir,
                "overfitRisk": q_overfit
            })
    except Exception as e:
        st.warning(f"[SPARQL查询错误] {ttl_file_path}: {e}")

    return question_list

# 扫描目录下的TTL文件
def scan_ttl_files(model_type):
    """根据模型类型扫描对应知识库目录"""
    directory = st.session_state.kb_paths.get(model_type, "")
    if not directory:
        st.error(f"未配置 {model_type} 知识库路径！")
        return []
    ttl_files = glob.glob(os.path.join(directory, "*.ttl"))
    return ttl_files

def add_new_rdf_file(file, model_type):
    """根据模型类型保存到对应目录"""
    target_dir = st.session_state.kb_paths.get(model_type, "")
    if not target_dir:
        st.error("请先配置知识库路径！")
        return None
    os.makedirs(target_dir, exist_ok=True)  # 确保目录存在
    file_path = os.path.join(target_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def delete_rdf_file(file_name, model_type):
    """根据模型类型删除文件"""
    target_dir = st.session_state.kb_paths.get(model_type, "")
    if not target_dir:
        return False
    file_path = os.path.join(target_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

def collect_rdf_data(selected_questions, model_type, ttl_directory):
    """收集选中问题相关的所有 RDF 三元组"""
    merged_graph = rdflib.Graph()
    label_counter = {}
    for item in selected_questions:
        file_path = os.path.join(ttl_directory, item["file"])
        g = rdflib.Graph()
        # g.parse(file_path, format="turtle")
        # 修改后（在 collect_rdf_data 函数中）
        g.parse(
            file_path,
            format="turtle",
            publicID="http://example.org/ontology#"  # 添加基础 URI
        )
        # 从文件名提取模型名称（去掉.ttl扩展名）
        model_name = os.path.splitext(item["file"])[0]  # 关键修复
        # 查询问题实体及其关联规则
        query = f"""
            PREFIX : <http://example.org/ontology#>
            CONSTRUCT {{
                ?qEntity ?p ?o .
                ?rule ?rule_p ?rule_o .
            }}
            WHERE {{
                ?rule :hasQuestion ?qEntity ;
                      ?rule_p ?rule_o .
                ?qEntity rdfs:label ?label ;
                         ?p ?o .
                FILTER (str(?label) = "{item['question']}")
            }}
        """
        # === 新增：处理重复标签 ===
        for s, p, o in g.query(query):
            # 处理问题实体的标签
            if p == RDFS.label and str(p) == "http://www.w3.org/2000/01/rdf-schema#label":
                original_label = str(o)
                # 更新计数器
                label_counter[original_label] = label_counter.get(original_label, 0) + 1
                count = label_counter[original_label]

                # 如果是重复标签则添加后缀
                new_label = original_label if count == 1 else f"{original_label} [{count}]"

                # 删除旧的三元组
                merged_graph.remove((s, p, o))
                # 添加新标签三元组
                merged_graph.add((s, p, Literal(new_label)))
            else:
                merged_graph.add((s, p, o))
    return merged_graph

# 定义核心命名空间
EX = Namespace("http://example.org/ontology#")
HYBRID = Namespace("http://example.org/hybrid/")

def generate_merged_ttl(merged_graph, model_names):
    # # 清理模型名称
    model_names_clean = [name.replace("Model", "").strip() for name in model_names]
    # ontology_uri = URIRef(f"http://example.org/{'_And_'.join(model_names_clean)}HybridModel")
    model_names_encoded = [quote(name.replace("Model", "").strip()) for name in model_names]
    ontology_uri = URIRef(f"http://example.org/merged/{'_and_'.join(model_names_encoded)}")
    # === 关键修复：绑定所有命名空间前缀 ===
    merged_graph.bind("ex", EX)
    merged_graph.bind("hybrid", HYBRID)
    merged_graph.bind("owl", OWL)
    merged_graph.bind("rdfs", RDFS)

    # 添加本体声明
    merged_graph.add((ontology_uri, RDF.type, OWL.Ontology))
    merged_graph.add((ontology_uri, EX.modelType, Literal("Hybrid")))

    # 添加核心属性（确保唯一性）
    core_props = [
        (EX.lowerBound, "数值范围的下界"),
        (EX.upperBound, "数值范围的上界"),
        (EX.hasTag, "问题实体的标签")
    ]
    for prop, comment in core_props:
        if (prop, RDF.type, RDF.Property) not in merged_graph:
            merged_graph.add((prop, RDF.type, RDF.Property))
            merged_graph.add((prop, RDFS.comment, Literal(comment)))

    # 收集所有唯一标签
    tags = {str(o) for _, _, o in merged_graph.triples((None, EX.hasTag, None))}

    # 生成跨模型比较规则
    for tag in tags:
        # 使用完整的URI代替前缀（避免依赖前缀绑定）
        cross_rule_uri = EX[f"CrossCompare_{tag}"]
        merged_graph.add((
            cross_rule_uri,
            RDF.type,
            EX.CrossModelComparisonRule
        ))
        merged_graph.add((
            cross_rule_uri,
            EX.comparedMetric,
            Literal(tag)
        ))
        # 添加模型列表
        for model in model_names_clean:
            merged_graph.add((
                cross_rule_uri,
                EX.comparedModels,
                Literal(model)
            ))
        # 添加优化方向
        optim_dir = "maximize" if "r2" in tag else "minimize"
        merged_graph.add((
            cross_rule_uri,
            EX.optimizationDirection,
            Literal(optim_dir)
        ))

    return merged_graph.serialize(format="turtle")

def map_rule_class_to_variable(rule_class_uri):
    rule_class = rdflib.URIRef(rule_class_uri)
    local_name = rule_class.split("#")[-1]
    if local_name.endswith("Rule"):
        variable = local_name[:-4].lower()
        return variable
    return "unknown_variable"

def get_model_required(g):
    """获取知识图谱的 modelRequired 属性"""
    query = """
    PREFIX : <http://example.org/ontology#>
    SELECT ?modelRequired
    WHERE {
        ?s a owl:Ontology ;
           :modelRequired ?modelRequired .
    }
    """
    results = list(g.query(query))
    return str(results[0][0]) if results else None

def get_question_optimization_direction(g, question_text):
    """获取特定问题的优化方向"""
    query = f"""
    PREFIX : <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?optimDir
    WHERE {{
        ?qEntity rdfs:label "{question_text}" ;
                 :optimizationDirection ?optimDir .
    }}
    """
    results = list(g.query(query))
    return str(results[0][0]) if results else None

# 生成模板
def generate_template(question, model_rules, optim_dir="neutral", overfit_risk=None, has_conflicts=False):
    if not model_rules:
        return None, f"# {question}\n\nNo rules found for this question."

    template_lines = []
    template_lines.append(question)
    template_lines.append("")  # 空行

    # 新增：标志变量，用于判断是否需要生成跨模型比较与建议
    has_r2_rules = False
    compare_data = []

    for model_rule in model_rules:
        model_name = model_rule["model_name"]
        model_display_name = model_rule["model_display_name"]
        rules = model_rule["rules"]

        first_rule = rules[0]
        rule_class = first_rule["rule_class"]
        variable = map_rule_class_to_variable(rule_class)

        unique_variable = f"{model_name}_{variable}"

        category = first_rule["rule_category"].lower().strip()

        if category == "constraint":
            placeholder = f"{{{{ {unique_variable} | round(3) }}}}"
            template_lines.append(
                f"The {variable.upper()} value of the {model_rule['model_display_name']} is {placeholder}.")
            template_lines.append("")

            sorted_rules = sorted([r for r in rules if r["lower_bound"] is not None],
                                  key=lambda r: float(r["lower_bound"]))

            for i, rule in enumerate(sorted_rules):
                lb = rule.get("lower_bound", 0.0)
                ub = rule.get("upper_bound", 0.0)
                description = rule["relationship_description"]
                recommendation = rule["recommendation"]

                condition = f"{unique_variable} >= {lb} and {unique_variable} < {ub}"

                if i == 0:
                    template_lines.append("{% if " + condition + " -%}")
                else:
                    template_lines.append("{% elif " + condition + " -%}")
                template_lines.append(f"The interpretation for {variable.upper()} is {description}. {recommendation}")

            last_rule = sorted_rules[-1]
            description = last_rule["relationship_description"]
            recommendation = last_rule["recommendation"]
            template_lines.append("{% else -%}")
            template_lines.append(f"The interpretation for {variable.upper()} is {description}. {recommendation}")
            template_lines.append("{% endif %}")
            template_lines.append("")  # 空行

        elif category == "descriptive":
            placeholder = f"{{{{ {unique_variable} }}}}"
            template_lines.append(f"The {variable.upper()} of the {model_rule['model_display_name']} is {placeholder}.")
            template_lines.append("")

            for rule in rules:
                description = rule["relationship_description"]
                recommendation = rule["recommendation"]

                if description.strip().lower().startswith(f"the {variable} is {{".lower()):
                    continue

                template_lines.append(f"{description}")
                template_lines.append("")

        else:
            placeholder = f"{{{{ {unique_variable} }}}}"
            template_lines.append(f"The {variable.upper()} value is {placeholder}.")
            template_lines.append("")

        compare_data.append({
            "var_name": unique_variable,
            "model_name": model_display_name
        })

    if len(compare_data) > 1:
        template_lines.append("\n{# --- Cross-model comparison --- #}")

        # 生成最佳值判断
        if optim_dir == "maximize":
            template_lines.append("{% set best_value = [" + ", ".join(
                c['var_name'] for c in compare_data
            ) + "] | max %}")
            compare_operator = ">="
        elif optim_dir == "minimize":
            template_lines.append("{% set best_value = [" + ", ".join(
                c['var_name'] for c in compare_data
            ) + "] | min %}")
            compare_operator = "<="
        else:  # neutral
            template_lines.append("{% set best_value = None %}")
            compare_operator = "=="

        # 生成模型比较条件链
        if optim_dir != "neutral":
            template_lines.append("\n")
            for i, model in enumerate(compare_data):
                condition = f"{model['var_name']} {compare_operator} best_value"
                if i == 0:
                    template_lines.append("{% if "+condition+" %}")
                else:
                    template_lines.append("{% elif "+condition+" %}")
                template_lines.append("The " + model["model_name"] + " has the best performance.")

            template_lines.append("{% endif %}\n")

        # 添加过拟合提示
        if overfit_risk:
            template_lines.append(f"\n{{# Overfitting Risk Note: #}}\n{overfit_risk}")

    # 在模板末尾添加冲突警告
    if has_conflicts:
        template_lines.append("\n{# --- Conflict Warning --- #}")
        template_lines.append("Knowledge graph files have conflicting answer rules for the same question under the same model. A detailed examination is recommended.")

    file_name = f"answer_{variable}.txt"
    template_content = "\n".join(template_lines)
    return file_name, template_content

# 查询问题
def search_question(query_text, ttl_files):
    results = defaultdict(list)
    for ttl_file in ttl_files:
        questions = load_questions_from_ttl(ttl_file)  # 现在返回字典列表
        for q in questions:
            # 同时匹配文本和标签
            if (query_text.lower() in q["text"].lower()) or \
               (query_text.lower() == q["tag"].lower()):
                results[ttl_file].append(q)  # 返回整个字典
    return results

# 修改RDF文件的函数
def modify_rdf_file(model_type, file_name, question, new_description=None, new_recommendation=None, new_lower_bound=None,
                    new_upper_bound=None):
    """
    修改 RDF 文件中的问题、答案描述以及上下边界。
    """
    ttl_file_path = os.path.join(st.session_state.kb_paths[model_type], file_name)

    g = rdflib.Graph()
    try:
        g.parse(ttl_file_path, format="turtle")
    except Exception as e:
        st.error(f"[解析错误] 无法解析 {ttl_file_path}: {e}")
        return False

    # SPARQL 查询，查找与问题相关的规则
    query = """
    PREFIX : <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?rule ?desc ?rec ?lb ?ub
    WHERE {
        ?rule :hasQuestion ?qEntity ;
              :relationshipDescription ?desc ;
              :recommendation ?rec ;
              :lowerBound ?lb ;
              :upperBound ?ub .
        ?qEntity rdfs:label ?label .
        FILTER (str(?label) = "%s")
    }
    """ % question.replace('"', '\\"')  # Escape quotes in question

    # 执行查询，获取规则
    rules_to_modify = g.query(query)
    if not rules_to_modify:
        st.warning(f"未找到与问题 '{question}' 相关的规则。")
        return False

    # 更新每个匹配的规则
    for row in rules_to_modify:
        rule = row[0]
        # 如果提供了新的描述，则修改描述
        if new_description:
            g.set((rule, rdflib.URIRef("http://example.org/ontology#relationshipDescription"),
                   rdflib.Literal(new_description)))
        # 如果提供了新的推荐，则修改推荐
        if new_recommendation:
            g.set(
                (rule, rdflib.URIRef("http://example.org/ontology#recommendation"), rdflib.Literal(new_recommendation)))
        # 如果提供了新的上下边界，则修改
        if new_lower_bound is not None:
            g.set((rule, rdflib.URIRef("http://example.org/ontology#lowerBound"), rdflib.Literal(new_lower_bound)))
        if new_upper_bound is not None:
            g.set((rule, rdflib.URIRef("http://example.org/ontology#upperBound"), rdflib.Literal(new_upper_bound)))

    # 保存修改后的 RDF 文件
    try:
        g.serialize(ttl_file_path, format="turtle")
        st.success(f"文件 {ttl_file_path} 修改成功！")
        return True
    except Exception as e:
        st.error(f"修改文件 {ttl_file_path} 失败: {e}")
        return False

def delete_rdf_file_question(ttl_file_path, question):
    """
    删除指定问题及其规则。
    """
    g = rdflib.Graph()
    try:
        g.parse(ttl_file_path, format="turtle")
    except Exception as e:
        st.error(f"[解析错误] 无法解析 {ttl_file_path}: {e}")
        return False

    query = """
    PREFIX : <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    DELETE WHERE {
        ?rule :hasQuestion ?qEntity .
        ?qEntity rdfs:label ?label .
        FILTER (str(?label) = "%s")
    }
    """ % question.replace('"', '\\"')

    try:
        g.update(query)
        g.serialize(ttl_file_path, format="turtle")
        st.success(f"问题 '{question}' 及相关规则已删除！")
        return True
    except Exception as e:
        st.error(f"删除问题 '{question}' 时发生错误: {e}")
        return False

def rename_question(g, old_question, new_question):
    # 找到与 old_question 匹配的问题实体
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?qEntity
    WHERE {
      ?qEntity rdfs:label ?label .
      FILTER (str(?label) = "%s")
    }
    """ % old_question.replace('"', '\\"')

    results = []
    for row in g.query(query):
        results.append(row[0])

    if not results:
        return False, "未找到要重命名的问题实体"

    qEntity = results[0]
    # 删除旧的 label
    g.remove((qEntity, rdflib.RDFS.label, rdflib.Literal(old_question)))
    # 添加新的 label
    g.add((qEntity, rdflib.RDFS.label, rdflib.Literal(new_question)))
    return True, "问题文本已更新"

def update_question_in_file(ttl_file_path, old_question, new_question=None, new_tag=None, new_optim_dir=None,
                          new_overfit_risk=None, new_rule_category=None, new_rule_class=None,
                          new_rule_desc=None, new_rule_recom=None, new_bounds_list=None):
    """
    修改已有问题的所有属性。
    :param ttl_file_path: RDF文件路径
    :param old_question: 原问题文本
    :param new_question: 新问题文本
    :param new_tag: 新标签
    :param new_optim_dir: 新优化方向
    :param new_overfit_risk: 新过拟合风险
    :param new_rule_category: 新规则类别
    :param new_rule_class: 新规则类
    :param new_rule_desc: 新规则描述
    :param new_rule_recom: 新规则推荐
    :param new_bounds_list: 新的边界列表
    :return: (bool, msg)
    """
    g = rdflib.Graph()
    try:
        g.parse(ttl_file_path, format="turtle")
    except Exception as e:
        return False, f"解析错误: {e}"

    # 找到问题实体
    query = """
    PREFIX : <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?qEntity ?tag ?optimDir ?overfitRisk
    WHERE {
      ?qEntity rdfs:label ?label .
      OPTIONAL { ?qEntity :hasTag ?tag . }
      OPTIONAL { ?qEntity :optimizationDirection ?optimDir . }
      OPTIONAL { ?qEntity :overfittingRisk ?overfitRisk . }
      FILTER (str(?label) = "%s")
    }
    """ % old_question.replace('"', '\\"')

    results = list(g.query(query))
    if not results:
        return False, f"未找到问题 '{old_question}'"

    q_entity = results[0][0]
    old_tag = results[0][1]
    old_optim_dir = results[0][2]
    old_overfit_risk = results[0][3]

    # 更新问题文本
    if new_question:
        g.remove((q_entity, RDFS.label, Literal(old_question)))
        g.add((q_entity, RDFS.label, Literal(new_question)))

    # 更新标签
    if new_tag is not None:
        if old_tag:
            g.remove((q_entity, EX.hasTag, old_tag))
        if new_tag:
            g.add((q_entity, EX.hasTag, Literal(new_tag)))

    # 更新优化方向
    if new_optim_dir is not None:
        if old_optim_dir:
            g.remove((q_entity, EX.optimizationDirection, old_optim_dir))
        if new_optim_dir:
            g.add((q_entity, EX.optimizationDirection, Literal(new_optim_dir)))

    # 更新过拟合风险
    if new_overfit_risk is not None:
        if old_overfit_risk:
            g.remove((q_entity, EX.overfittingRisk, old_overfit_risk))
        if new_overfit_risk:
            g.add((q_entity, EX.overfittingRisk, Literal(new_overfit_risk)))

    # 更新规则
    if new_rule_category or new_rule_class or new_rule_desc or new_rule_recom or new_bounds_list:
        # 找到所有相关规则
        rule_query = """
        PREFIX : <http://example.org/ontology#>
        SELECT ?rule ?category ?class ?desc ?recom ?lb ?ub
        WHERE {
          ?rule :hasQuestion ?qEntity ;
                :ruleCategory ?category ;
                :ruleClass ?class ;
                :relationshipDescription ?desc ;
                :recommendation ?recom .
          OPTIONAL { ?rule :lowerBound ?lb . }
          OPTIONAL { ?rule :upperBound ?ub . }
          FILTER(?qEntity = <%s>)
        }
        """ % q_entity

        rules = list(g.query(rule_query))
        
        # 删除所有旧规则
        for rule, _, _, _, _, _, _ in rules:
            g.remove((rule, None, None))
            g.remove((None, None, rule))

        # 创建新规则
        if new_bounds_list:
            for bounds in new_bounds_list:
                new_rule = BNode()
                g.add((new_rule, RDF.type, EX[new_rule_class or "MSERule"]))
                g.add((new_rule, EX.hasQuestion, q_entity))
                g.add((new_rule, EX.ruleCategory, Literal(new_rule_category or "Descriptive")))
                g.add((new_rule, EX.ruleClass, EX[new_rule_class or "MSERule"]))
                if new_rule_desc:
                    g.add((new_rule, EX.relationshipDescription, Literal(new_rule_desc)))
                if new_rule_recom:
                    g.add((new_rule, EX.recommendation, Literal(new_rule_recom)))
                if bounds.get('lower_bound'):
                    g.add((new_rule, EX.lowerBound, Literal(float(bounds['lower_bound']))))
                if bounds.get('upper_bound'):
                    g.add((new_rule, EX.upperBound, Literal(float(bounds['upper_bound']))))
        else:
            # 如果没有新的边界列表，创建一个默认规则
            new_rule = BNode()
            g.add((new_rule, RDF.type, EX[new_rule_class or "MSERule"]))
            g.add((new_rule, EX.hasQuestion, q_entity))
            g.add((new_rule, EX.ruleCategory, Literal(new_rule_category or "Descriptive")))
            g.add((new_rule, EX.ruleClass, EX[new_rule_class or "MSERule"]))
            if new_rule_desc:
                g.add((new_rule, EX.relationshipDescription, Literal(new_rule_desc)))
            if new_rule_recom:
                g.add((new_rule, EX.recommendation, Literal(new_rule_recom)))

    try:
        g.serialize(destination=ttl_file_path, format="turtle")
        return True, "问题及其规则已成功更新"
    except Exception as e:
        return False, f"保存错误: {e}"

def delete_question(g, question_text):
    # 找到问题实体
    query_q = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?qEntity
    WHERE {
      ?qEntity rdfs:label ?label .
      FILTER (str(?label) = "%s")
    }
    """ % question_text.replace('"','\\"')

    results = list(g.query(query_q))
    if not results:
        return False, "未找到需要删除的问题"

    q_entity = results[0][0]
    # 删除所有引用q_entity的三元组
    g.remove((q_entity, None, None))
    g.remove((None, None, q_entity))
    # 这样就彻底删除了这个问题实体
    return True, "问题已删除"

def add_question_to_file(ttl_file_path, q_text, tag, optim_dir, overfit_risk=None, rule_category=None, 
                        rule_class=None, rule_desc=None, rule_recom=None, bounds_list=None):
    # 添加参数校验
    if not all([q_text, tag, optim_dir]):
        return False, "问题文本、标签和优化方向为必填项"
    """
    在指定 RDF 文件里，新增一个问题以及对应的一条或多条rule三元组。
    :param ttl_file_path: 目标RDF文件
    :param q_text: 新的问题文本
    :param tag: 问题标签
    :param optim_dir: 优化方向
    :param overfit_risk: 过拟合风险
    :param rule_category: 规则类别（Constraint/Descriptive）
    :param rule_class: 规则类型（如R2Rule）
    :param rule_desc: 规则描述
    :param rule_recom: 规则推荐
    :param bounds_list: 边界列表，每个元素是一个字典，包含lower_bound和upper_bound
    :return: (bool, msg)
    """
    g = rdflib.Graph()
    try:
        g.parse(ttl_file_path, format="turtle")
    except Exception as e:
        return False, f"解析错误: {e}"

    # 1) 生成一个新的question实体
    question_entity = BNode()
    g.add((question_entity, rdflib.RDFS.label, Literal(q_text)))
    g.add((question_entity, EX.hasTag, Literal(tag)))
    g.add((question_entity, EX.optimizationDirection, Literal(optim_dir)))
    if overfit_risk:
        g.add((question_entity, EX.overfittingRisk, Literal(overfit_risk)))

    # 2) 生成规则
    if rule_category == "Constraint":
        # 为每组边界创建一个规则
        for i, bounds in enumerate(bounds_list):
            new_rule = BNode()
            g.add((new_rule, rdflib.RDF.type, URIRef(f"http://example.org/ontology#{rule_class}")))
            g.add((new_rule, EX.hasQuestion, question_entity))
            g.add((new_rule, EX.ruleCategory, Literal(rule_category)))
            
            # 添加边界
            if bounds.get('lower_bound') is not None:
                g.add((new_rule, EX.lowerBound, Literal(float(bounds['lower_bound']))))
            if bounds.get('upper_bound') is not None:
                g.add((new_rule, EX.upperBound, Literal(float(bounds['upper_bound']))))
            
            # 添加描述和推荐
            if rule_desc:
                g.add((new_rule, EX.relationshipDescription, Literal(rule_desc)))
            if rule_recom:
                g.add((new_rule, EX.recommendation, Literal(rule_recom)))
    else:
        # 对于Descriptive规则，只需要一个规则
        new_rule = BNode()
        g.add((new_rule, rdflib.RDF.type, URIRef(f"http://example.org/ontology#{rule_class}")))
        g.add((new_rule, EX.hasQuestion, question_entity))
        g.add((new_rule, EX.ruleCategory, Literal(rule_category)))
        
        if rule_desc:
            g.add((new_rule, EX.relationshipDescription, Literal(rule_desc)))
        if rule_recom:
            g.add((new_rule, EX.recommendation, Literal(rule_recom)))

    try:
        g.serialize(destination=ttl_file_path, format="turtle")
        return True, f"已添加新问题'{q_text}'"
    except Exception as e:
        return False, f"保存错误: {e}"

def delete_question_in_file(ttl_file_path, question_text):
    """
    从RDF文件删除指定问题（含关联rule或仅删除question实体? 视需求）
    """
    g = rdflib.Graph()
    try:
        g.parse(ttl_file_path, format="turtle")
    except Exception as e:
        return False, f"解析错误: {e}"

    # 找到 question entity
    query_q = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?qEntity
    WHERE {
      ?qEntity rdfs:label ?label .
      FILTER (str(?label) = "%s")
    }
    """ % question_text.replace('"','\\"')

    results = list(g.query(query_q))
    if not results:
        return False, f"未找到问题 '{question_text}'"

    q_entity = results[0][0]

    # 找到所有 rule 关联到该q_entity
    query_rule = """
    PREFIX : <http://example.org/ontology#>
    SELECT ?rule
    WHERE {
      ?rule :hasQuestion ?qe .
      FILTER(?qe = <%s>)
    }
    """ % q_entity

    for rowr in g.query(query_rule):
        rule_uri = rowr[0]
        # 删除rule相关三元组
        g.remove((rule_uri, None, None))
        g.remove((None, None, rule_uri))

    # 最后删除q_entity自己
    g.remove((q_entity, None, None))
    g.remove((None, None, q_entity))

    try:
        g.serialize(destination=ttl_file_path, format="turtle")
        return True, f"已删除问题 '{question_text}' 及其相关rule"
    except Exception as e:
        return False, f"保存错误: {e}"

def map_question_to_variable(question_uri, g):
    """通过问题的 hasTag 属性获取变量名"""
    query = f"""
        PREFIX : <http://example.org/ontology#>
        SELECT ?tag
        WHERE {{
            <{question_uri}> :hasTag ?tag .
        }}
    """
    result = g.query(query)
    return str(result.bindings[0]['tag']) if result else None

def truncate_filename(filename, max_length=20):
    """截断文件名，保留扩展名"""
    if len(filename) <= max_length:
        return filename
    name, ext = os.path.splitext(filename)
    return f"{name[:max_length-3]}...{ext}"

def build_kb_tree():
    """构建知识图谱库的树状结构"""
    kb_tree = {}
    kb_types = {
        "regression": "回归模型知识库",
        "classification": "分类模型知识库",
        "math": "数学模型知识库",
        "hybrid": "混合模型知识库"
    }
    
    has_valid_path = False
    for kb_type, display_name in kb_types.items():
        path = st.session_state.kb_paths.get(kb_type, "")
        if path and os.path.exists(path):
            has_valid_path = True
            ttl_files = glob.glob(os.path.join(path, "*.ttl"))
            # 存储完整文件名和显示用的截断文件名
            kb_tree[display_name] = [
                {
                    "full_name": os.path.splitext(os.path.basename(f))[0],
                    "display_name": truncate_filename(os.path.splitext(os.path.basename(f))[0])
                }
                for f in ttl_files
            ]
    
    return kb_tree, has_valid_path

def compare_rules_boundaries(rules1, rules2):
    """
    比较两组规则的边界组数，返回边界组数较多的规则组
    :param rules1: 第一组规则
    :param rules2: 第二组规则
    :return: 边界组数较多的规则组
    """
    # 计算每组规则中的边界组数
    def count_boundary_groups(rules):
        count = 0
        for rule in rules:
            if rule.get("lower_bound") is not None or rule.get("upper_bound") is not None:
                count += 1
        return count
    
    count1 = count_boundary_groups(rules1)
    count2 = count_boundary_groups(rules2)
    
    # 返回边界组数较多的规则组
    if count1 >= count2:
        return rules1
    else:
        return rules2

# 主函数
def main():
    st.set_page_config(page_title="RDF Models Viewer", layout="wide")
    
    # 从配置文件加载路径
    if "kb_paths" not in st.session_state:
        st.session_state.kb_paths = load_config()
    
    # 初始化选中的文件和当前选中的知识库类型
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "生成模板"
    if "selected_kb_type" not in st.session_state:
        st.session_state.selected_kb_type = None
        
    # 检查配置是否有效
    if not all(st.session_state.kb_paths.values()):
        st.error("配置文件中的知识库路径配置不完整，请检查 config.json 文件！")
        return

    # 检查所有路径是否存在
    missing_paths = [path for path in st.session_state.kb_paths.values() if not os.path.exists(path)]
    if missing_paths:
        st.error(f"以下路径不存在: {', '.join(missing_paths)}")
        return

    # 创建三列布局
    left_col, middle_col, right_col = st.columns([1, 3, 1])

    with left_col:
        st.markdown("### 知识图谱库")
        st.markdown("---")
        kb_tree, has_valid_path = build_kb_tree()
        if not has_valid_path:
            st.warning("未读取到知识图谱库")
        else:
            # 清空之前的选择
            if st.button("清除选择"):
                st.session_state.selected_files = []
                st.session_state.selected_kb_type = None
                st.rerun()
            
            for kb_type, files in kb_tree.items():
                with st.expander(f"📁 {kb_type}", expanded=True):
                    # 检查是否有其他类型被选中
                    is_other_type_selected = (st.session_state.selected_kb_type is not None and 
                                            st.session_state.selected_kb_type != kb_type)
                    
                    for file_info in files:
                        # 创建唯一的key
                        checkbox_key = f"select_{kb_type}_{file_info['full_name']}"
                        # 检查是否已选中
                        is_selected = file_info['full_name'] in st.session_state.selected_files
                        
                        # 使用checkbox进行选择，如果其他类型被选中则禁用
                        checkbox = st.checkbox(
                            f"📄 {file_info['display_name']}", 
                            key=checkbox_key,
                            value=is_selected,
                            help=file_info['full_name'],
                            disabled=is_other_type_selected
                        )
                        
                        if checkbox and not is_other_type_selected:
                            if file_info['full_name'] not in st.session_state.selected_files:
                                st.session_state.selected_files.append(file_info['full_name'])
                                st.session_state.selected_kb_type = kb_type
                                st.session_state.active_tab = "生成模板"
                                st.rerun()
                        elif not checkbox and file_info['full_name'] in st.session_state.selected_files:
                            st.session_state.selected_files.remove(file_info['full_name'])
                            if not st.session_state.selected_files:
                                st.session_state.selected_kb_type = None
                            st.rerun()
                    
                    # 如果当前类型有选中的文件，显示提示信息
                    if is_other_type_selected:
                        st.info(f"请先取消选择 {st.session_state.selected_kb_type} 中的文件")

    with middle_col:
        st.title("RDF Models Viewer - Knowledge Base Configuration")
        # 添加顶部菜单（移除查询问题选项）
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["生成模板", "增加RDF文件", "删除RDF文件", "修改RDF文件", "修改配置"])

        with tab1:
            # 显示知识库类型选择
            model_type = st.selectbox(
                "选择知识库类型",
                options=["regression", "classification", "math", "hybrid"],
                format_func=lambda x: {
                    "regression": "回归模型知识库",
                    "classification": "分类模型知识库",
                    "math": "数学模型知识库",
                    "hybrid": "混合模型知识库"
                }[x],
                key="gen_template_model_type"
            )

            # 获取对应知识库路径
            ttl_directory = st.session_state.kb_paths.get(model_type, "")
            if not ttl_directory or not os.path.exists(ttl_directory):
                st.error(f"路径 '{ttl_directory}' 未配置或不存在，请检查知识库配置！")
                st.stop()

            # 扫描目录下的TTL文件
            all_ttl_paths = scan_ttl_files(model_type)
            if not all_ttl_paths:
                st.error(f"在目录 {ttl_directory} 中未找到任何 .ttl 文件")
                st.stop()

            # 构建文件-问题字典
            rdf_files_dict = {}
            for path in all_ttl_paths:
                file_name = os.path.basename(path)
                questions = load_questions_from_ttl(path)
                rdf_files_dict[file_name] = questions

            # 创建两列布局
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # 使用session_state中的选中文件作为默认值
                selected_files = st.multiselect(
                    "选择模型", 
                    options=list(rdf_files_dict.keys()), 
                    default=[f"{f}.ttl" for f in st.session_state.selected_files if f in [os.path.splitext(k)[0] for k in rdf_files_dict.keys()]]
                )
                st.markdown("---")

                selected_questions = []
                for file_name in selected_files:
                    questions = rdf_files_dict[file_name]
                    if questions:
                        st.subheader(file_name)
                        for q_dict in questions:  # q_dict 是包含 text 和 tag 的字典
                            question_text = q_dict["text"]
                            checkbox_id = f"{file_name}_{question_text}"
                            if st.checkbox(question_text, key=checkbox_id):
                                selected_questions.append({"file": file_name, "question": question_text})

                generate = st.button("生成问答模板")

            with col2:
                if selected_files:
                    st.write("### 已选择的模型及其问题")
                    for file_name in selected_files:
                        st.write(f"**{file_name}**")
                        questions = rdf_files_dict[file_name]
                        for q_dict in questions:
                            question_text = q_dict["text"]
                            if {"file": file_name, "question": question_text} in selected_questions:
                                st.markdown(f"- [x] {question_text}")
                            else:
                                st.markdown(f"- [ ] {question_text}")
                else:
                    st.write("### 请在左侧选择一个或多个模型，并选择相应的问题。")

                if generate:
                    if not selected_questions:
                        st.warning("请至少选择一个问题以生成问答模板。")
                    else:
                        st.write("### 生成中...")
                        grouped_questions = defaultdict(list)
                        for item in selected_questions:
                            grouped_questions[item["question"]].append(item["file"])

                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                            for question, files in grouped_questions.items():
                                model_rules = []
                                optim_dir = "neutral"  # 默认值
                                overfit_risk = None
                                has_conflicts = False

                                # 检查模型要求和优化方向
                                if len(files) > 1:
                                    model_requirements = {}
                                    optimization_directions = {}
                                    
                                    for file_name in files:
                                        # 确保文件名包含.ttl扩展名
                                        if not file_name.endswith('.ttl'):
                                            file_name = f"{file_name}.ttl"
                                        ttl_path = os.path.join(ttl_directory, file_name)
                                        g = rdflib.Graph()
                                        try:
                                            g.parse(ttl_path, format="turtle")
                                            # 获取模型要求
                                            model_req = get_model_required(g)
                                            if model_req:
                                                model_requirements[file_name] = model_req
                                            
                                            # 获取优化方向
                                            optim = get_question_optimization_direction(g, question)
                                            if optim:
                                                optimization_directions[file_name] = optim
                                        except Exception as e:
                                            st.error(f"[解析错误] 无法解析 {file_name}: {e}")
                                            continue

                                    # 检查模型要求是否相同
                                    unique_model_reqs = set(model_requirements.values())
                                    if len(unique_model_reqs) == 1:
                                        # 如果模型要求相同，检查优化方向
                                        unique_optim_dirs = set(optimization_directions.values())
                                        if len(unique_optim_dirs) > 1:
                                            has_conflicts = True
                                        else:
                                            # 如果优化方向也相同，比较规则边界组数
                                            if len(files) == 2:
                                                rules1 = model_rules[0]["rules"]
                                                rules2 = model_rules[1]["rules"]
                                                # 保留边界组数较多的规则
                                                model_rules[0]["rules"] = compare_rules_boundaries(rules1, rules2)
                                                model_rules[1]["rules"] = model_rules[0]["rules"]

                                # 从第一个文件获取优化方向和过拟合风险
                                if files:
                                    first_file = files[0]
                                    # 确保文件名包含.ttl扩展名
                                    if not first_file.endswith('.ttl'):
                                        first_file = f"{first_file}.ttl"
                                    ttl_path = os.path.join(ttl_directory, first_file)
                                    g = rdflib.Graph()
                                    g.parse(ttl_path, format="turtle")

                                    # 查询该问题的优化属性
                                    query = f"""
                                        PREFIX : <http://example.org/ontology#>
                                        SELECT ?optimDir ?overfitRisk
                                        WHERE {{
                                            ?qEntity rdfs:label "{question}" ;
                                                     :optimizationDirection ?optimDir ;
                                                     :overfittingRisk ?overfitRisk .
                                        }}
                                    """
                                    result = g.query(query)
                                    if result:
                                        row = result.bindings[0]
                                        optim_dir = str(row.get('optimDir', 'neutral'))
                                        overfit_risk = str(row.get('overfitRisk', ''))
                                
                                for file_name in files:
                                    # 确保文件名包含.ttl扩展名
                                    if not file_name.endswith('.ttl'):
                                        file_name = f"{file_name}.ttl"
                                    ttl_path = os.path.join(ttl_directory, file_name)
                                    g = rdflib.Graph()
                                    try:
                                        g.parse(ttl_path, format="turtle")
                                    except Exception as e:
                                        st.error(f"[解析错误] 无法解析 {file_name}: {e}")
                                        continue
                                    query = """
                                    PREFIX : <http://example.org/ontology#>
                                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                    SELECT ?ruleClass ?lowerBound ?upperBound ?relationshipDescription ?recommendation ?ruleCategory
                                    WHERE {
                                      ?rule a ?ruleClass ;
                                            :hasQuestion ?qEntity ;
                                            :relationshipDescription ?relationshipDescription ;
                                            :recommendation ?recommendation ;
                                            :ruleCategory ?ruleCategory .
                                      OPTIONAL { ?rule :lowerBound ?lowerBound . }
                                      OPTIONAL { ?rule :upperBound ?upperBound . }
                                      ?qEntity rdfs:label ?label ;
                                                   :hasTag ?tag .
                                      FILTER (str(?label) = "%s")
                                    }
                                    """ % question.replace('"', '\\"')
                                    rules = []

                                    try:
                                        for row in g.query(query):
                                            rule_class = str(row.ruleClass).split("#")[-1]
                                            lower_bound = row.lowerBound.toPython() if row.lowerBound else None
                                            upper_bound = row.upperBound.toPython() if row.upperBound else None
                                            relationship_description = str(row.relationshipDescription)
                                            recommendation = str(row.recommendation)
                                            rule_category = str(row.ruleCategory).strip('"') if row.ruleCategory else "none"

                                            rules.append({
                                                "rule_class": rule_class,
                                                "lower_bound": lower_bound,
                                                "upper_bound": upper_bound,
                                                "relationship_description": relationship_description,
                                                "recommendation": recommendation,
                                                "rule_category": rule_category
                                            })
                                    except Exception as e:
                                        st.warning(f"[SPARQL查询错误] {file_name}: {e}")

                                    if rules:
                                        # 去掉扩展名，如 .ttl
                                        base_no_ext, _ = os.path.splitext(file_name)
                                        # 进一步替换 Model => model, Regression => Regression
                                        model_display_name = base_no_ext.replace('Model', ' model').replace('Regression',
                                                                                                            ' regression')
                                        # model_name = model_display_name.lower().replace(' ', '_')
                                        model_name = base_no_ext
                                        model_rules.append({
                                            "model_name": model_name,
                                            "model_display_name": model_display_name,
                                            "rules": rules
                                        })

                                if model_rules:
                                    # 检查模型要求和优化方向
                                    if len(model_rules) > 1:
                                        model_requirements = {}
                                        optimization_directions = {}
                                        
                                        for model_rule in model_rules:
                                            file_name = model_rule["model_name"]
                                            # 确保文件名包含.ttl扩展名
                                            if not file_name.endswith('.ttl'):
                                                file_name = f"{file_name}.ttl"
                                            ttl_path = os.path.join(ttl_directory, file_name)
                                            g = rdflib.Graph()
                                            try:
                                                g.parse(ttl_path, format="turtle")
                                                # 获取模型要求
                                                model_req = get_model_required(g)
                                                if model_req:
                                                    model_requirements[file_name] = model_req
                                                
                                                # 获取优化方向
                                                optim = get_question_optimization_direction(g, question)
                                                if optim:
                                                    optimization_directions[file_name] = optim
                                            except Exception as e:
                                                st.error(f"[解析错误] 无法解析 {file_name}: {e}")
                                                continue

                                        # 检查模型要求是否相同
                                        unique_model_reqs = set(model_requirements.values())
                                        if len(unique_model_reqs) == 1:
                                            # 如果模型要求相同，检查优化方向
                                            unique_optim_dirs = set(optimization_directions.values())
                                            if len(unique_optim_dirs) > 1:
                                                has_conflicts = True
                                            else:
                                                # 如果优化方向也相同，比较规则边界组数
                                                if len(model_rules) == 2:
                                                    rules1 = model_rules[0]["rules"]
                                                    rules2 = model_rules[1]["rules"]
                                                    # 保留边界组数较多的规则
                                                    model_rules[0]["rules"] = compare_rules_boundaries(rules1, rules2)
                                                    model_rules[1]["rules"] = model_rules[0]["rules"]

                                    file_out_name, template_content = generate_template(
                                        question,
                                        model_rules,
                                        optim_dir=optim_dir,
                                        overfit_risk=overfit_risk,
                                        has_conflicts=has_conflicts
                                    )
                                    if not file_out_name:
                                        st.warning(f"未生成模板文件，因为未找到规则。问题：{question}")
                                        continue

                                    zip_file.writestr(file_out_name, template_content)

                            # 收集跨模型数据
                            # merged_graph = collect_rdf_data(selected_questions, model_type, ttl_directory)
                            # 修改后（确保使用正确的知识库路径）
                            kb_type_map = {
                                "regression": st.session_state.kb_paths["regression"],
                                "classification": st.session_state.kb_paths["classification"],
                                "math": st.session_state.kb_paths["math"],
                                "hybrid": st.session_state.kb_paths["hybrid"]
                            }
                            ttl_directory = kb_type_map[model_type]
                            merged_graph = collect_rdf_data(selected_questions, model_type,ttl_directory)
                            merged_ttl = generate_merged_ttl(merged_graph, [os.path.splitext(f)[0] for f in selected_files])

                            # 写入整合图谱
                            model_names_clean = [os.path.splitext(f)[0] for f in selected_files]  # 移除.ttl扩展名
                            filename = f"Integrated_{'_And_'.join(model_names_clean)}.ttl"
                            zip_file.writestr(filename, merged_ttl)
                            # 生成脚本
                            script_lines = []
                            script_lines.append("import pandas as pd")
                            script_lines.append("import DataScienceComponents as DC")
                            script_lines.append("import NLGComponents as NC")
                            script_lines.append("from dash import html")
                            script_lines.append("from jinja2 import Environment, BaseLoader")
                            # 脚本主体

                            script_lines.append("# 加载数据（例如从CSV文件）")
                            script_lines.append("data = pd.read_csv('some_data.csv')")
                            script_lines.append("Xcol = ['col1', 'col2',...]")
                            script_lines.append("ycol = 'target'")
                            # 在选择问题和模型后，初始化 variables_to_fill 字典
                            variables_to_fill = {}
                            question_to_variable_map = {
                                "Is the relationship between the variables strong?": "r2",
                                "Is the MAPE value acceptable?": "mape",
                                "What is the MAE of this model?": "mae",
                                "What is the RMSE of this model?": "rmse",
                                "What is the MSE of this model?": "mse"
                            }

                            # 记录已调用的模型
                            called_models = set()  # 用于追踪已经调用过的模型
                            # 根据选择的问题和模型，调用数据科学组件进行模型拟合
                            for item in selected_questions:
                                file_name = item["file"]
                                question = item["question"]
                                # 根据选择的问题获取对应的变量
                                var = question_to_variable_map.get(question)
                                # 根据文件名判断模型类型，并生成对应的代码
                                if file_name == "LinearRegressionModel.ttl":
                                    # 如果没有调用过线性回归模型的拟合函数，则调用
                                    if "linear_regression" not in called_models:
                                        script_lines.append(
                                            "linear_model_results = DC.ModelFitting().LinearSKDefaultModel(data, Xcol, ycol)")
                                        called_models.add("linear_regression")  # 标记线性回归模型已调用

                                elif file_name == "GradientBoostingRegressionModel.ttl":
                                    # 如果没有调用过GB模型的拟合函数，则调用
                                    if "gradient_boosting" not in called_models:
                                        script_lines.append(
                                            "gradient_boosting_results = DC.ModelFitting().GradientBoostingDefaultModel(data, Xcol, ycol)")
                                        called_models.add("gradient_boosting")  # 标记GB模型已调用

                            script_lines.append("env = Environment(loader=BaseLoader())")
                            script_lines.append("app, listTabs = NC.start_app()")
                            script_lines.append("QA = template_name.render(variable=variable)")
                            script_lines.append("children = [html.P(QA)]")
                            script_lines.append("NC.dash_tab_add(listTabs, label, children)")

                            script_lines.append("NC.run_app(app, listTabs, portnum=8050)")

                            full_script = "\n".join(script_lines)
                            zip_file.writestr("auto_generated_pipeline.py", full_script)

                        zip_buffer.seek(0)
                        st.success("问答模板生成成功！")
                        st.download_button(label="下载问答模板 ZIP", data=zip_buffer, file_name="qa_templates.zip",
                                           mime="application/zip")

        with tab2:
            model_type = st.selectbox("选择目标知识库类型", ["regression", "classification", "math", "hybrid"])
            uploaded_file = st.file_uploader("上传新的RDF文件", type=["ttl"])
            if uploaded_file is not None:
                file_path = add_new_rdf_file(uploaded_file, model_type)
                st.success(f"文件 {uploaded_file.name} 上传成功，保存在 {file_path}")

        with tab3:
            model_type = st.selectbox("选择知识库类型", ["regression", "classification", "math", "hybrid"])
            ttl_files = scan_ttl_files(model_type)
            file_to_delete = st.selectbox("选择要删除的文件", [os.path.basename(f) for f in ttl_files])
            if st.button("删除"):
                success = delete_rdf_file(file_to_delete, model_type)
                if success:
                    st.success(f"文件 {file_to_delete} 已成功删除！")
                else:
                    st.error(f"未能找到文件 {file_to_delete}，删除失败！")

        with tab4:
            # 选择知识库类型
            model_type = st.selectbox(
                "选择知识库类型",
                options=["regression", "classification", "math", "hybrid"],
                format_func=lambda x: {
                    "regression": "回归模型知识库",
                    "classification": "分类模型知识库",
                    "math": "数学模型知识库",
                    "hybrid": "混合模型知识库"
                }[x]
            )
            ttl_directory = st.session_state.kb_paths.get(model_type, "")
            if not ttl_directory:
                st.error(f"未配置 {model_type} 路径！")
                st.stop()

            ttl_files = scan_ttl_files(model_type)
            ttl_files_names = [os.path.basename(f) for f in ttl_files]

            # 选择要修改的文件
            selected_file = st.selectbox("选择要修改的RDF文件", ["请选择文件"] + ttl_files_names)
            if selected_file == "请选择文件":
                st.warning("请先选择要修改的RDF文件。")
            else:
                # 已选择文件
                file_path = os.path.join(ttl_directory, selected_file)

                st.write(f"### 您选择的文件：{selected_file}")

                # 读取该文件内的问题
                questions_in_file = load_questions_from_ttl(file_path)

                # 构建下拉列表, 添加一个"添加新问题"选项
                choice_list = ["添加新问题"] + questions_in_file
                selected_question = st.selectbox("请选择要操作的问题，或添加新问题", choice_list)

                if selected_question == "添加新问题":
                    # 用户想添加新问题
                    st.write("#### 添加新问题")

                    # 基本信息部分
                    st.write("##### 基本信息")
                    new_q_text = st.text_input("新问题文本")
                    new_tag = st.text_input("问题标签")
                    new_optim_dir = st.selectbox("优化方向", ["maximize", "minimize", "neutral"])
                    new_overfit_risk = st.text_area("过拟合风险说明(可选)")

                    # 规则信息部分
                    st.write("##### 规则信息")
                    new_rule_category = st.selectbox("规则类别", ["Constraint", "Descriptive"])
                    new_rule_class = st.text_input("规则类型(如R2Rule)")
                    new_rule_desc = st.text_area("规则描述")
                    new_rule_recom = st.text_area("规则推荐")

                    # 边界信息部分
                    st.write("##### 边界信息")
                    if "bounds_list" not in st.session_state:
                        st.session_state.bounds_list = [{"lower_bound": "", "upper_bound": ""}]

                    for i, bounds in enumerate(st.session_state.bounds_list):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            bounds["lower_bound"] = st.text_input(f"下界 {i+1}", bounds["lower_bound"])
                        with col2:
                            bounds["upper_bound"] = st.text_input(f"上界 {i+1}", bounds["upper_bound"])
                        with col3:
                            if st.button("删除", key=f"del_{i}"):
                                st.session_state.bounds_list.pop(i)
                                st.rerun()

                    if st.button("添加新的边界组"):
                        st.session_state.bounds_list.append({"lower_bound": "", "upper_bound": ""})
                        st.rerun()

                    if st.button("确认修改"):
                        # 处理空值
                        new_optim_dir = None if new_optim_dir == "" else new_optim_dir
                        new_rule_category = None if new_rule_category == "" else new_rule_category
                        new_rule_class = None if new_rule_class == "" else new_rule_class

                        # 过滤掉空的边界组
                        bounds_list = [b for b in st.session_state.bounds_list if b["lower_bound"] or b["upper_bound"]]

                        if not new_q_text:
                            st.error("问题文本不能为空！")
                            return

                        # 调用添加函数而不是修改函数
                        success, msg = add_question_to_file(
                            ttl_file_path=file_path,
                            q_text=new_q_text,
                            tag=new_tag,
                            optim_dir=new_optim_dir if new_optim_dir else "neutral",
                            overfit_risk=new_overfit_risk if new_overfit_risk else None,
                            rule_category=new_rule_category if new_rule_category else "Descriptive",
                            rule_class=new_rule_class if new_rule_class else "BaseRule",
                            rule_desc=new_rule_desc,
                            rule_recom=new_rule_recom,
                            bounds_list=[b for b in st.session_state.bounds_list if
                                         b["lower_bound"] or b["upper_bound"]]
                        )

                        if success:
                            st.success(msg)
                            st.session_state.bounds_list = [{"lower_bound": "", "upper_bound": ""}]
                        else:
                            st.error(msg)

                else:
                    # 用户选择了文件中已存在的一个问题
                    st.write(f"#### 您选择修改/删除的问题：{selected_question}")
                    # 让用户选择要进行的操作
                    operation = st.radio("请选择操作", ["修改该问题", "删除该问题"])

                    if operation == "修改该问题":
                        # 提供可修改项
                        st.write("##### 基本信息")
                        new_q_text = st.text_input("新的问题文本(留空表示不改)", "")
                        new_tag = st.text_input("新的标签(留空表示不改)", "")
                        new_optim_dir = st.selectbox("新的优化方向", 
                                                    ["", "maximize", "minimize", "neutral"],
                                                    format_func=lambda x: {"": "保持不变", "maximize": "最大化", "minimize": "最小化", "neutral": "中性"}[x])
                        new_overfit_risk = st.text_area("新的过拟合风险说明(留空表示不改)", "")

                        st.write("##### 规则信息")
                        new_rule_category = st.selectbox("新的规则类别",
                                                        ["", "Constraint", "Descriptive"],
                                                        format_func=lambda x: {"": "保持不变", "Constraint": "约束规则", "Descriptive": "描述性规则"}[x])
                        new_rule_class = st.selectbox("新的规则类",
                                                     ["", "MSERule", "R2Rule", "MAERule", "RMSERule", "MAPERule"],
                                                     format_func=lambda x: {"": "保持不变", "MSERule": "MSE规则", "R2Rule": "R2规则", 
                                                                          "MAERule": "MAE规则", "RMSERule": "RMSE规则", 
                                                                          "MAPERule": "MAPE规则"}[x])
                        new_rule_desc = st.text_area("新的规则描述(留空表示不改)", "")
                        new_rule_recom = st.text_area("新的规则推荐(留空表示不改)", "")

                        # 边界信息部分
                        st.write("##### 边界信息")
                        if "bounds_list" not in st.session_state:
                            st.session_state.bounds_list = [{"lower_bound": "", "upper_bound": ""}]

                        for i, bounds in enumerate(st.session_state.bounds_list):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                bounds["lower_bound"] = st.text_input(f"下界 {i+1}", bounds["lower_bound"])
                            with col2:
                                bounds["upper_bound"] = st.text_input(f"上界 {i+1}", bounds["upper_bound"])
                            with col3:
                                if st.button("删除", key=f"del_{i}"):
                                    st.session_state.bounds_list.pop(i)
                                    st.rerun()

                        if st.button("添加新的边界组"):
                            st.session_state.bounds_list.append({"lower_bound": "", "upper_bound": ""})
                            st.rerun()

                        if st.button("确认修改"):
                            # 处理空值
                            new_optim_dir = None if new_optim_dir == "" else new_optim_dir
                            new_rule_category = None if new_rule_category == "" else new_rule_category
                            new_rule_class = None if new_rule_class == "" else new_rule_class
                            
                            # 过滤掉空的边界组
                            bounds_list = [b for b in st.session_state.bounds_list if b["lower_bound"] or b["upper_bound"]]

                            success, msg = update_question_in_file(
                                file_path,
                                old_question=selected_question["text"],  # 修复后
                                new_question=new_q_text or None,
                                new_tag=new_tag or None,
                                new_optim_dir=new_optim_dir,
                                new_overfit_risk=new_overfit_risk or None,
                                new_rule_category=new_rule_category,
                                new_rule_class=new_rule_class,
                                new_rule_desc=new_rule_desc or None,
                                new_rule_recom=new_rule_recom or None,
                                new_bounds_list=bounds_list if bounds_list else None
                            )
                            if success:
                                st.success(msg)
                                # 清空边界列表
                                st.session_state.bounds_list = [{"lower_bound": "", "upper_bound": ""}]
                            else:
                                st.error(msg)

                    elif operation == "删除该问题":
                        if st.button("确认删除"):
                            success, msg = delete_question_in_file(file_path, selected_question["text"])

                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)

        with tab5:
            st.header("修改知识图谱库配置")
            st.write("### 当前配置")
            
            # 显示当前配置
            current_config = st.session_state.kb_paths
            for kb_type, path in current_config.items():
                st.text_input(
                    f"{kb_type} 知识库路径",
                    value=path,
                    key=f"config_{kb_type}",
                    disabled=True
                )
            
            st.write("### 修改配置")
            # 创建新的配置字典
            new_config = {}
            
            # 为每个知识库类型创建文件夹选择
            kb_types = {
                "regression": "回归模型知识图谱库",
                "classification": "分类模型知识图谱库",
                "math": "数学模型知识图谱库",
                "hybrid": "混合模型知识图谱库"
            }
            
            # 使用列布局并排显示选择框
            cols = st.columns(2)
            for idx, (key, label) in enumerate(kb_types.items()):
                with cols[idx % 2]:
                    st.write(f"**{label}**")
                    # 使用文本输入框显示和编辑路径
                    new_path = st.text_input(
                        "输入路径",
                        value=current_config.get(key, ""),
                        key=f"new_{key}",
                        help="请输入完整的文件夹路径，例如：D:\\study\\ADSTPapp\\kg\\kgr"
                    )
                    # 验证路径是否存在
                    if new_path and not os.path.exists(new_path):
                        st.warning(f"警告：路径 '{new_path}' 不存在！")
                    
                    new_config[key] = new_path
            
            # 添加保存按钮
            if st.button("保存配置"):
                # 验证所有路径是否都已填写
                if not all(new_config.values()):
                    st.error("请填写所有知识库路径！")
                else:
                    # 验证所有路径是否存在
                    missing_paths = [path for path in new_config.values() if not os.path.exists(path)]
                    if missing_paths:
                        st.error(f"以下路径不存在：{', '.join(missing_paths)}")
                    else:
                        try:
                            # 更新配置文件
                            config_data = {"kb_paths": new_config}
                            with open('config.json', 'w', encoding='utf-8') as f:
                                json.dump(config_data, f, indent=4, ensure_ascii=False)
                            
                            # 更新session_state中的路径
                            st.session_state.kb_paths = new_config
                            
                            st.success("配置已成功保存！")
                            st.rerun()  # 刷新页面以应用新配置
                        except Exception as e:
                            st.error(f"保存配置时出错：{str(e)}")

    with right_col:
        st.markdown("### 问题查询")
        st.markdown("---")
        # 收集所有知识库路径下的文件
        all_ttl_paths = []
        for kb_type in ["regression", "classification", "math", "hybrid"]:
            dir_path = st.session_state.kb_paths.get(kb_type, "")
            if dir_path and os.path.exists(dir_path):
                all_ttl_paths.extend(glob.glob(os.path.join(dir_path, "*.ttl")))

        search_query = st.text_input("输入查询关键词", "")
        if search_query:
            search_results = search_question(search_query, all_ttl_paths)

            if search_results:
                st.markdown("#### 搜索结果")
                for file_path, questions in search_results.items():
                    file_name = os.path.basename(file_path)
                    with st.expander(f"📄 {file_name}", expanded=False):
                        for q_dict in questions:
                            st.markdown(f"- **问题**: {q_dict['text']}")
                            if q_dict['tag']:
                                st.markdown(f"  - 标签: {q_dict['tag']}")
                            if q_dict['optim_dir']:
                                st.markdown(f"  - 优化方向: {q_dict['optim_dir']}")
                            if q_dict['overfitRisk']:
                                st.markdown(f"  - 过拟合风险: {q_dict['overfitRisk']}")
                            st.markdown("---")
            else:
                st.warning(f"未找到与 '{search_query}' 相关的问题。")

if __name__ == "__main__":
    main() 