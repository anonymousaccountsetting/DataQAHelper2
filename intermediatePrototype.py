import streamlit as st
import pandas as pd
import re
import os
from LocalLLMcomponents import SettingForOllama
from datasciencecomponents import DataScienceRegressionComponents,DataScienceClassifierComponents,DataEngineering,FindBestModel
from NLGcomponents import RegressionTemplateBasedTextGeneration,ClassifierTemplateBasedTextGeneration,SettingForChatGPT,AutoFindBestModel,LoadQuestionBank
from LLMcomponents import SettingForLLM
import glob

# 从字符串中提取第一个整数，用于解析匹配结果
def extract_first_integer(string):
    match = re.search(r"\d+", string)
    if match:
        return int(match.group())
    return 0

# 初始化 LLM 与组件
set_for_localLLM = SettingForOllama()
set_for_GPT = SettingForChatGPT()
ds_data_engineering = DataEngineering()
ds_regression = DataScienceRegressionComponents()
ds_classifier = DataScienceClassifierComponents()
nlg_reg = RegressionTemplateBasedTextGeneration()
nlg_cls = ClassifierTemplateBasedTextGeneration()
loader = LoadQuestionBank()
set_for_LLM=SettingForLLM()



def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_step = 1
        st.session_state.df = None
        st.session_state.dependent_var = None
        st.session_state.independent_vars = []
        st.session_state.background_text = ''
        st.session_state.llm_bg = []
        st.session_state.llm_messages = []
        st.session_state.user_questions = []
        st.session_state.recommended_models = []
        st.session_state.reg_choice = '无'
        st.session_state.cls_choice = '无'
        st.session_state.selected_model = None
        st.session_state.fit_results = {}
        st.session_state.single_question = ''
        st.session_state.single_answer = ''
        st.session_state.llm_mode = 'ollama'
        st.session_state.openai_model = 'gpt-3.5-turbo'
        st.session_state.openai_key = ''

        # 自动加载题库文件夹中的所有 txt 文件
        base_path = r".\apptemplates\QuestionBank\modelquestionbanks"
        txt_files = glob.glob(os.path.join(base_path, "*.txt"))
        st.session_state.model_question_bank_paths = txt_files

        # 加载题库内容
        banks = {}
        for path in txt_files:
            model_name = os.path.splitext(os.path.basename(path))[0].replace('_questions', '')
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                banks[model_name] = content
                st.write(f"已加载 {os.path.basename(path)}...")
            except Exception as e:
                banks[model_name] = ''
                st.warning(f"无法加载题库 {model_name}: {e}")

        st.session_state.model_question_banks = banks
        st.session_state.model_match_counts = {name: 0 for name in banks.keys()}
        st.session_state.recommended_models = []


# 统计用户问题与各模型题库的匹配次数
def compute_model_match_counts(user_questions):
    counts = {name: 0 for name in st.session_state.model_question_banks.keys()}
    # 确保背景已发送给 LLM
    # set_for_localLLM.set_chat_background([], st.session_state.background_text)  # 如需启用对话上下文
    if st.session_state.llm_mode == "ollama":
        for q in user_questions:
            for model_name, full_text in st.session_state.model_question_banks.items():
                if not full_text:
                    continue
                output = set_for_localLLM.question_matching(q, full_text)
                if extract_first_integer(output) != 0:
                    counts[model_name] += 1
    else:
        for q in user_questions:
            for model_name, full_text in st.session_state.model_question_banks.items():
                if not full_text:
                    continue

                st.session_state.payload, st.session_state.llm_messages = set_for_LLM.set_payload(q, st.session_state.openai_chatmodel, st.session_state.llm_messages)
                output, st.session_state.llm_messages = set_for_LLM.send_response_receive_output(st.session_state.openai_url, st.session_state.openai_headers, st.session_state.payload, st.session_state.llm_messages)

                if extract_first_integer(output) != 0:
                    counts[model_name] += 1
    return counts

# 步骤1：数据上传
def step_upload_data():
    st.header("📤 步骤1：数据上传")
    uploaded_file = st.file_uploader(
        "上传CSV文件", type=["csv"], help="支持标准CSV格式，最大文件大小200MB"
    )
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ 文件上传成功！")
            st.session_state.data_preview_expanded = True
        except Exception as e:
            st.error(f"❌ 文件读取错误: {e}")
            st.session_state.df = None
    if st.session_state.df is not None:
        with st.expander("数据预览", expanded=st.session_state.data_preview_expanded):
            st.dataframe(st.session_state.df, height=300, use_container_width=True)
            st.subheader("数据摘要")
            c1, c2, c3 = st.columns(3)
            c1.metric("总行数", len(st.session_state.df))
            c2.metric("变量数", len(st.session_state.df.columns))
            c3.metric("缺失值", st.session_state.df.isna().sum().sum())
    c1, c2 = st.columns([1,1])
    with c1:
        if st.session_state.df is not None and st.button("重新上传"):
            st.session_state.df = None
            st.rerun()
    with c2:
        if st.session_state.df is not None and st.button("下一步 →", type="primary"):
            st.session_state.current_step = 2
            st.rerun()
# --- 新增部分：让用户选择题库文件夹 ---
    st.divider()
    st.subheader("📁 可选：更换问题库文件夹")

    custom_qb_path = st.text_input("请输入新的问题库文件夹路径：", value=r".\apptemplates\QuestionBank\modelquestionbanks")

    if st.button("加载新的问题库路径"):
        import glob, os
        if os.path.isdir(custom_qb_path):
            txt_files = glob.glob(os.path.join(custom_qb_path, "*.txt"))
            if not txt_files:
                st.warning("⚠️ 当前文件夹下未找到任何 .txt 问题库文件。")
            else:
                banks = {}
                for path in txt_files:
                    model_name = os.path.splitext(os.path.basename(path))[0].replace('_questions', '')
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            banks[model_name] = f.read()
                        st.write(f"已加载：{os.path.basename(path)}")
                    except Exception as e:
                        banks[model_name] = ''
                        st.warning(f"⚠️ 加载 {path} 失败: {e}")
                st.session_state.model_question_banks = banks
                st.session_state.model_match_counts = {k: 0 for k in banks.keys()}
                st.success(f"✅ 成功加载 {len(txt_files)} 个题库文件。")
        else:
            st.error("❌ 输入的路径不是一个有效文件夹。")




# 步骤2：变量选择
def step_select_variables():
    st.header("📊 步骤2：变量选择")
    if st.session_state.df is None:
        st.warning("请先上传数据文件")
        st.session_state.current_step = 1
        st.rerun()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("目标变量 (Y)")
        st.session_state.dependent_var = st.selectbox("选择因变量", st.session_state.df.columns)
    with col2:
        st.subheader("特征变量 (X)")
        avail = [c for c in st.session_state.df.columns if c != st.session_state.dependent_var]
        st.session_state.independent_vars = st.multiselect("选择自变量", avail, default=avail[:3])
    if len(st.session_state.independent_vars) < 1:
        st.error("请至少选择一个自变量")
        return
    prev, _, nxt = st.columns([1,8,1])
    with prev:
        if st.button("← 上一步"):
            st.session_state.current_step = 1
            st.rerun()
    with nxt:
        if st.button("下一步 →", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

# 步骤3：确认变量选择
def step_confirm_selection():
    st.header("✅ 步骤3：选择确认")

    st.subheader("当前选择")
    st.markdown(f"""
    <div style="padding:15px; border-radius:10px; background:#f0f2f6">
    🔍 **分析目标**：预测/分析 {st.session_state.dependent_var}
    🛠️ **使用特征**：{', '.join(st.session_state.independent_vars)}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("变量详细信息", expanded=True):
        tab1, tab2 = st.tabs(["目标变量分析", "特征变量概览"])

        with tab1:
            st.write(st.session_state.df[st.session_state.dependent_var].describe())
            if pd.api.types.is_numeric_dtype(st.session_state.df[st.session_state.dependent_var]):
                st.line_chart(st.session_state.df[st.session_state.dependent_var])
            else:
                st.bar_chart(st.session_state.df[st.session_state.dependent_var].value_counts())

        with tab2:
            st.dataframe(st.session_state.df[st.session_state.independent_vars].describe())

    st.divider()
    st.subheader("🤖 选择大语言模型来源")

    llm_source = st.radio(
        "请选择用于问题匹配与回答的LLM模型来源：",
        options=["使用本地 Ollama", "调用 OpenAI API"],
        index=0,
        key="llm_source_choice"
    )

    if llm_source == "调用 OpenAI API":
        st.session_state.llm_mode = "api"
        st.text_input("请输入模型名称", key="openai_model")
        st.text_input("请输入 OpenAI API Key", type="password", key="openai_key")
    else:
        st.session_state.llm_mode = "ollama"

    col_prev, col_mid, col_next = st.columns([1, 8, 1])
    with col_prev:
        if st.button("← 重新选择"):
            st.session_state.current_step = 2
            st.rerun()

    with col_next:
        if st.button("开始建模 →", type="primary"):
            st.session_state.current_step = 4
            st.rerun()

# 步骤4：输入数据集背景知识
def step_input_background():

    st.header("🌐 步骤4：输入数据集背景知识")
    st.markdown("请提供关于数据集的背景信息，以便 LLM 在问题匹配时参考。可简述数据来源、含义、预处理等。比如：\n- 数据收集于...\n- 包含变量...\n- 数据已完成缺失值填充等。")

    text = st.text_area("请输入背景知识：", height=200, value=st.session_state.background_text)

    if text:
        st.session_state.background_text = text
        st.success("✅ 背景信息已保存！")

        if st.session_state.llm_mode == "ollama":
            # 使用本地 LLM
            messages, bg = set_for_localLLM.set_chat_background([], text)
            st.session_state.llm_bg = bg
            st.session_state.llm_messages = messages
        else:
            # 使用 OpenAI API
            key = st.session_state.openai_key
            model = st.session_state.openai_model
            url, bg, chatmodel, headers, messages = set_for_LLM.set_chatGPT(text, key, model)

            st.session_state.llm_bg = bg
            st.session_state.llm_messages = messages
            st.session_state.openai_url = url
            st.session_state.openai_headers = headers
            st.session_state.openai_chatmodel = chatmodel

    prev, _, nxt = st.columns([1, 8, 1])
    with prev:
        if st.button("← 返回确认变量"):
            st.session_state.current_step = 3
            st.rerun()
    with nxt:
        if st.button("下一步 →", type="primary"):
            if not st.session_state.background_text:
                st.error("请先输入背景知识！")
            else:
                st.session_state.current_step = 5
                st.rerun()


# 步骤5：批量输入问题并推荐模型
def step_input_questions():
    st.header("❓ 步骤5：批量输入问题")
    with st.expander("格式说明", expanded=True):
        st.markdown("1. 问题一\n2. 问题二\n3. 问题三")
    text = st.text_area("请输入问题列表:", height=200)
    if text:
        parts = re.split(r"\n\s*\d+\.\s*", text.strip())
        st.session_state.user_questions = [p.strip() for p in parts if p.strip()]
    if st.session_state.user_questions:
        st.success("解析到以下问题：")
        for i, q in enumerate(st.session_state.user_questions, 1): st.write(f"{i}. {q}")
        counts = compute_model_match_counts(st.session_state.user_questions)
        st.session_state.model_match_counts = counts
        max_cnt = max(counts.values()) if counts else 0
        recommended = [m for m,c in counts.items() if c==max_cnt and c>0]
        st.session_state.recommended_models = recommended
        st.subheader("🔍 匹配统计")
        for m,c in counts.items(): st.write(f"- {m}: {c}")
        if recommended:
            st.success(f"⭐ 推荐：{', '.join(recommended)}")
        else:
            st.warning("⚠️ 未匹配到合适模型")



    prev,mid,nxt = st.columns([1,1,1])
    with prev:
        if st.button("← 返回背景"): st.session_state.current_step=4; st.rerun()
    with mid:
        if st.button("跳过推荐"): st.session_state.current_step=6; st.rerun()
    with nxt:
        if st.button("下一步 →"):
            if not st.session_state.recommended_models:
                st.error("请先完成模型匹配！")
            else:
                st.session_state.current_step=6; st.rerun()

# 步骤6：选择并拟合模型
def step_select_model():
    st.header("⚙️ 步骤6：选择并拟合模型")
    reg_opts = ["Statsmodels Linear Regression","Scikit-learn Linear Regression","Ridge Regression","Gradient Boosting Regression","Random Forest Regression","Lasso Regression","Bayesian Ridge Regression","Auto-fit Best Regression"]
    cls_opts = ["Logistic Regression","Linear Discriminant Analysis","SVM - Linear Kernel","Ridge Classifier","Random Forest Classifier","Decision Tree Classifier","Auto-fit Best Classifier"]
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("回归")
        st.session_state.reg_choice = st.radio("选择一个回归模型", ["无"]+reg_opts)
        if st.session_state.reg_choice=="Ridge Regression": st.number_input("Ridge Alpha", key='ridge_alpha')
        if st.session_state.reg_choice=="Lasso Regression": st.number_input("Lasso Alpha", key='lasso_alpha')
        if st.session_state.reg_choice=="Bayesian Ridge Regression":
            st.number_input("Alpha1", key='bayes_a1')
            st.number_input("Alpha2", key='bayes_a2')
            st.number_input("Lambda1", key='bayes_l1')
            st.number_input("Lambda2", key='bayes_l2')
        if st.session_state.reg_choice=="Auto-fit Best Regression": st.text_input("Criterion", key='reg_crit')
    with c2:
        st.subheader("分类")
        st.session_state.cls_choice = st.radio("选择一个分类模型", ["无"]+cls_opts)
        if st.session_state.cls_choice=="Auto-fit Best Classifier": st.text_input("Criterion", key='cls_crit')
    if st.button("下一步 →"):
        r, c = st.session_state.reg_choice, st.session_state.cls_choice
        if (r!="无")==(c!="无"):
            st.error("只能选择一个模型！"); return
        df = st.session_state.df
        X = df[st.session_state.independent_vars]
        y = df[st.session_state.dependent_var]
        results = {}
        if r!="无":
            # 数据清洗
            df_clean = ds_data_engineering.clean_data(df, 0.8, Xcol=st.session_state.independent_vars, ycol=st.session_state.dependent_var)
            Xc = df_clean[st.session_state.independent_vars]
            yc = df_clean[st.session_state.dependent_var]
            # 拟合回归
            if r=="Statsmodels Linear Regression": coef,pval,r2 = ds_regression.train_sm_linear_regression(Xc,yc)
            elif r=="Scikit-learn Linear Regression": coef,pval,r2 = ds_regression.train_sk_linear_regression(Xc,yc)
            elif r=="Ridge Regression": coef,pval,r2 = ds_regression.train_ridge_regression(Xc,yc,st.session_state.ridge_alpha)
            elif r=="Lasso Regression": coef,pval,r2 = ds_regression.train_lasso_regression(Xc,yc,st.session_state.lasso_alpha)
            elif r=="Bayesian Ridge Regression": coef,pval,r2 = ds_regression.train_BayesianRidge_regression(
                Xc,yc,st.session_state.bayes_a1,st.session_state.bayes_a2,st.session_state.bayes_l1,st.session_state.bayes_l2)
            elif r=="Gradient Boosting Regression": coef,pval,r2,train_r2,test_r2 = ds_regression.train_gradient_boosting_regression(Xc,yc); results.update({'train_r2':train_r2,'test_r2':test_r2})
            elif r=="Random Forest Regression": coef,pval,r2,train_r2,test_r2 = ds_regression.train_random_forest_regression(Xc,yc); results.update({'train_r2':train_r2,'test_r2':test_r2})
            elif r=="Auto-fit Best Regression": name,detail,crit,_ = ds_data_engineering.find_best_regression(Xc,yc,st.session_state.dependent_var,st.session_state.reg_crit,st.session_state.independent_vars); results.update({'model_name':name,'detail':detail,'criterion':crit})
            # 生成系数与P值表
            coef_df = ds_regression.coefficients_with_Pvalues(coef,pval,st.session_state.independent_vars)
            results.update({'coeff_df':coef_df,'r_squared':r2})
            st.session_state.selected_model = r
            st.session_state.fit_results = results
        else:
            # 拟合分类
            if c=="Logistic Regression": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_logistic_regression(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="Linear Discriminant Analysis": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_linear_discriminant_analysis(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="SVM - Linear Kernel": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_SVC_classifier(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="Ridge Classifier": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_ridge_classifier(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="Random Forest Classifier": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_random_forest_classifier(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="Decision Tree Classifier": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_decision_tree_classifier(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            else: name,detail,crit,_ = ds_data_engineering.find_best_classifier(X,y,st.session_state.dependent_var,st.session_state.cls_crit,st.session_state.independent_vars); results.update({'model_name':name,'detail':detail,'criterion':crit}); model=None; coeff_df=None; acc=None; train_acc=None; test_acc=None
            results.update({'coeff_df':coeff_df,'accuracy':acc,'train_accuracy':train_acc,'test_accuracy':test_acc})
            st.session_state.selected_model = c
            st.session_state.fit_results = results
        st.session_state.current_step = 7
        st.rerun()

# 步骤7：单题提问
def step_input_single_question():
    st.header("❓ 步骤7：单题提问")
    st.session_state.single_question = st.text_input("输入您的问题：", value="", key='q_input')
    if st.session_state.single_question and st.button("提交问题 →"):
        q = st.session_state.single_question
        # 匹配模板节号
        if 'Regression' in st.session_state.selected_model:
            content = loader.load_regression_questions()
        else:
            content = loader.load_classifier_questions()
        if st.session_state.llm_mode == "ollama":
            sec = extract_first_integer(set_for_localLLM.question_matching(q, content))
        else:
            st.session_state.payload, st.session_state.llm_messages = set_for_LLM.set_payload(q,
                                                                                              st.session_state.openai_chatmodel,
                                                                                              st.session_state.llm_messages)
            output, st.session_state.llm_messages = set_for_LLM.send_response_receive_output(
                st.session_state.openai_url, st.session_state.openai_headers, st.session_state.payload,
                st.session_state.llm_messages)

            sec = extract_first_integer(output)
        # 获取模板问答
        if 'Regression' in st.session_state.selected_model:
            coef_df = st.session_state.fit_results.get('coeff_df')
            if sec==1:
                qs,ans = nlg_reg.Q_and_A_about_R2(coef_df, st.session_state.dependent_var, st.session_state.selected_model, st.session_state.fit_results.get('r_squared'))
            elif sec==2:
                qs,ans = nlg_reg.Q_and_A_about_coefficients(coef_df, st.session_state.dependent_var)
            elif sec==3:
                qs,ans = nlg_reg.Q_and_A_about_importance(st.session_state.independent_vars, st.session_state.dependent_var, st.session_state.fit_results.get('coeff'), coef_df)
            elif sec==4:
                qs,ans = nlg_reg.Q_and_A_about_pvalues(coef_df, st.session_state.dependent_var)
            elif sec==5:
                qs,ans = nlg_reg.Q_and_A_about_ML_importance(st.session_state.independent_vars, st.session_state.dependent_var, coef_df)
            elif sec==6:
                qs,ans = nlg_reg.Q_and_A_about_ML_overfit(st.session_state.fit_results.get('train_r2'), st.session_state.dependent_var, st.session_state.fit_results.get('test_r2'))
            else:
                qs,ans = [],[]
        else:
            coeff_df = st.session_state.fit_results.get('coeff_df')
            if sec==1:
                qs,ans = nlg_cls.Q_and_A_about_accuracy(st.session_state.fit_results.get('accuracy'), st.session_state.selected_model)
            elif sec==2:
                qs,ans = nlg_cls.Q_and_A_about_coefficients(coeff_df, st.session_state.dependent_var)
            elif sec==3:
                qs,ans = nlg_cls.Q_and_A_about_importance(st.session_state.fit_results.get('coeff'), st.session_state.dependent_var, coeff_df, st.session_state.fit_results.get('model').classes_)
            elif sec==4:
                qs,ans = nlg_cls.Q_and_A_about_ML_overfit(st.session_state.fit_results.get('train_accuracy'), st.session_state.fit_results.get('test_accuracy'))
            elif sec==5:
                qs,ans = nlg_cls.Q_and_A_about_ML_importance(coeff_df, st.session_state.dependent_var)
            else:
                qs,ans = [],[]
        # 基于模板回答更新
        # default_answer = set_for_GPT.answer_update(q, qs, ans)

        default_answer="The user's question is:"+q+"\nThe corresponding predefined template questions are: "+qs+"\nSo now, please answer the user's question based on the analysis results and the above content."

        if st.session_state.llm_mode == "ollama":
            out, msgs = set_for_localLLM.send_response_receive_output(default_answer, st.session_state.llm_bg, st.session_state.llm_messages)
            st.session_state.llm_messages = msgs
            st.session_state.single_answer = out
        else:
            st.session_state.payload, st.session_state.llm_messages = set_for_LLM.set_payload(q,
                                                                                              st.session_state.openai_chatmodel,
                                                                                              st.session_state.llm_messages)
            out, msgs = set_for_LLM.send_response_receive_output(
                st.session_state.openai_url, st.session_state.openai_headers, st.session_state.payload,
                st.session_state.llm_messages)

            st.session_state.llm_messages = msgs
            st.session_state.single_answer = out
        st.session_state.current_step = 8
        st.rerun()

# 步骤8：显示回答并可再次提问
def step_show_single_answer():
     st.header("💬 步骤8：回答结果")
     # 显示前一步的回答
     st.write(st.session_state.single_answer)
     # 按钮栏：再次提问 或 生成总结
     col_reask, col_summary = st.columns([1,1])
     with col_reask:
         if st.button("再次提问"):
             st.session_state.current_step = 7
             st.rerun()
     with col_summary:
         if st.button("生成总结"):
             # 构造总结提示语
             summary_prompt = (
                 "Please summarize all the above Q&A interactions into a concise report focusing on the user's questions "
                 "about the dataset and the chosen model, without explaining the process."
             )
             # 调用 LLM 生成总结
             if st.session_state.llm_mode == "ollama":
                 out, msgs = set_for_localLLM.send_response_receive_output(
                     summary_prompt,
                     st.session_state.llm_bg,
                     st.session_state.llm_messages
                 )
                 # 更新消息历史并保存 summary
                 st.session_state.llm_messages = msgs
                 st.session_state.summary = out
             else:
                 st.session_state.payload, st.session_state.llm_messages = set_for_LLM.set_payload(summary_prompt,
                                                                                                   st.session_state.openai_chatmodel,
                                                                                                   st.session_state.llm_messages)
                 out, msgs = set_for_LLM.send_response_receive_output(
                     st.session_state.openai_url, st.session_state.openai_headers, st.session_state.payload,
                     st.session_state.llm_messages)
                 # 更新消息历史并保存 summary
                 st.session_state.llm_messages = msgs
                 st.session_state.summary = out
             # 在界面显示总结
             st.subheader("📄 总结报告")
             st.write(st.session_state.summary)
             # 将聊天记录保存为 docx，通知用户
             set_for_localLLM.save_chat_history_to_docx(st.session_state.llm_messages)
             st.success("✅ 报告已生成并保存在服务器根目录下的 data_report.docx")

 # 主流程
def main():
     st.set_page_config(page_title="智能数据分析助手", layout="wide")
     initialize_session_state()
     steps=["上传","变量","确认","背景","批量问","模型选择","单题问答","答案展示"]
     st.progress((st.session_state.current_step-1)/len(steps))
     if st.session_state.current_step==1: step_upload_data()
     elif st.session_state.current_step==2: step_select_variables()
     elif st.session_state.current_step==3: step_confirm_selection()
     elif st.session_state.current_step==4: step_input_background()
     elif st.session_state.current_step==5: step_input_questions()
     elif st.session_state.current_step==6: step_select_model()
     elif st.session_state.current_step==7: step_input_single_question()
     elif st.session_state.current_step==8: step_show_single_answer()

if __name__=="__main__":
     main()

