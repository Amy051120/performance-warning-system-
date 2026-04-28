"""
业绩变脸早期预警系统 - Streamlit Cloud演示版
完整版本,展示所有核心功能和交互
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 页面配置
st.set_page_config(
    page_title="业绩变脸早期预警系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# 模拟数据生成函数
@st.cache_data
def generate_sample_data():
    """生成模拟的股票数据"""
    np.random.seed(42)
    
    # 生成股票列表
    stocks = [
        ('600519', '贵州茅台'), ('000858', '五粮液'), ('002594', '比亚迪'),
        ('600036', '招商银行'), ('601318', '中国平安'), ('000001', '平安银行'),
        ('600000', '浦发银行'), ('000002', '万科A'), ('600276', '恒瑞医药'),
        ('000333', '美的集团'), ('600030', '中信证券'), ('601166', '兴业银行'),
        ('600887', '伊利股份'), ('000651', '格力电器'), ('601398', '工商银行'),
        ('601288', '农业银行'), ('600016', '民生银行'), ('601988', '中国银行'),
        ('600050', '中国联通'), ('601628', '中国人寿')
    ]
    
    # 生成历史数据
    data = []
    for code, name in stocks:
        for year in range(2012, 2026):
            # 随机生成预告类型
            forecast_types = ['大增', '略增', '扭亏', '续盈', '略降', '续亏', '不确定']
            forecast = np.random.choice(forecast_types)
            
            # 随机生成变脸标签(2012-2021年有标签,2022-2025年未知)
            if year <= 2021:
                face_change = np.random.choice([0, 1], p=[0.96, 0.04])
            else:
                face_change = -1  # 未知
            
            # 生成变脸概率(2022-2025年)
            if year >= 2022:
                prob = np.random.beta(2, 5)  # 大部分概率较低
            else:
                prob = np.nan
            
            data.append({
                'StockCode': code,
                'StockName': name,
                'Year': year,
                'ForecFinReportType': forecast,
                'face_change': face_change,
                'probability': prob
            })
    
    return pd.DataFrame(data)

# 加载数据
sample_df = generate_sample_data()

# 侧边栏导航
st.sidebar.title("📊 功能导航")
page = st.sidebar.radio("", [
    "🏠 系统概览",
    "🔍 股票预警查询",
    "🔮 2025年预测结果",
    "📈 模型性能",
    "🔬 消融实验",
    "🎯 预测示例",
    "📖 技术文档",
    "ℹ️ 关于系统"
])

# ==================== 页面1: 系统概览 ====================
if page == "🏠 系统概览":
    st.markdown('<h1 class="main-header">业绩变脸早期预警系统</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 系统简介
    st.markdown('<h2 class="sub-header">系统简介</h2>', unsafe_allow_html=True)
    st.markdown("""
    本系统基于**多模态特征融合**和**滚动预测框架**,实现上市公司业绩变脸的早期预警。
    
    **核心特点:**
    - 🎯 **多模态特征**: 财务指标 + 业绩预告数值 + BERT文本语义
    - 📊 **滚动预测**: 严格避免未来信息泄露
    - 🔬 **消融实验**: 量化各特征组合贡献
    - 💡 **可解释性**: SHAP特征重要性分析
    """)
    
    # 核心指标
    st.markdown('<h2 class="sub-header">核心数据指标</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 总样本数", "26,998", "2012-2025年")
    with col2:
        st.metric("⚠️ 变脸样本", "1,011", "占比3.74%")
    with col3:
        st.metric("🎯 特征维度", "121维", "三类特征融合")
    with col4:
        st.metric("📈 最优AUC", "0.7336", "随机森林")
    
    # 滚动预测框架
    st.markdown('<h2 class="sub-header">滚动预测框架</h2>', unsafe_allow_html=True)
    st.markdown("采用3年训练窗口预测下一年度,严格避免未来信息泄露:")
    
    cols = st.columns(4)
    windows = [
        ("第1轮", "2019-2021", "2022", "0.7280"),
        ("第2轮", "2020-2022", "2023", "0.7836"),
        ("第3轮", "2021-2023", "2024", "0.7655"),
        ("第4轮", "2022-2024", "2025", "0.7424"),
    ]
    for i, (rn, train, pred, auc) in enumerate(windows):
        with cols[i]:
            st.info(f"**{rn}**\n\n训练: {train}\n预测: {pred}\nAUC: {auc}")
    
    # 变脸定义
    st.markdown('<h2 class="sub-header">变脸三级定义</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error("**严重变脸**\n\n方向反转\n(正→负/负→正)")
    with col2:
        st.warning("**中度变脸**\n\n修正类型变化\n(大增→略增)")
    with col3:
        st.success("**轻度变脸**\n\n数值偏离 > 30%")

# ==================== 页面2: 模型性能 ====================
elif page == "📈 模型性能":
    st.markdown('<h1 class="main-header">模型性能分析</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 模型对比
    st.markdown('<h2 class="sub-header">模型性能对比</h2>', unsafe_allow_html=True)
    st.markdown("基于4轮滚动预测(2022-2025年)的平均性能:")
    
    performance_data = {
        '模型': ['随机森林', 'XGBoost', 'LightGBM'],
        'AUC': [0.7336, 0.7044, 0.7105],
        'Recall': [0.6137, 0.4927, 0.2869],
        'F1': [0.0851, 0.0793, 0.0958],
        'Precision': [0.0458, 0.0433, 0.0577]
    }
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df.style.format({
        'AUC': '{:.4f}',
        'Recall': '{:.4f}',
        'F1': '{:.4f}',
        'Precision': '{:.4f}'
    }), use_container_width=True, hide_index=True)
    
    st.success("✅ **随机森林模型表现最优**: AUC=0.7336, Recall=0.6137")
    
    # 详细性能
    st.markdown('<h2 class="sub-header">各轮预测详细结果</h2>', unsafe_allow_html=True)
    
    detail_data = {
        '轮次': ['Round1(2022)', 'Round2(2023)', 'Round3(2024)', 'Round4(2025)'],
        '训练样本': [8102, 8468, 8297, 8583],
        '测试样本': [2705, 2886, 2811, 2972],
        '变脸样本': [64, 56, 68, 42],
        'RF-AUC': [0.6870, 0.7783, 0.7227, 0.7465],
        'XGB-AUC': [0.6572, 0.7350, 0.7102, 0.7152],
        'LGB-AUC': [0.6954, 0.7262, 0.7183, 0.7022]
    }
    detail_df = pd.DataFrame(detail_data)
    st.dataframe(detail_df.style.format({
        'RF-AUC': '{:.4f}',
        'XGB-AUC': '{:.4f}',
        'LGB-AUC': '{:.4f}'
    }), use_container_width=True, hide_index=True)
    
    # 性能可视化
    st.markdown('<h2 class="sub-header">性能趋势图</h2>', unsafe_allow_html=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # AUC趋势
    years = [2022, 2023, 2024, 2025]
    rf_auc = [0.6870, 0.7783, 0.7227, 0.7465]
    xgb_auc = [0.6572, 0.7350, 0.7102, 0.7152]
    lgb_auc = [0.6954, 0.7262, 0.7183, 0.7022]
    
    ax1.plot(years, rf_auc, 'o-', label='Random Forest', linewidth=2, markersize=8)
    ax1.plot(years, xgb_auc, 's-', label='XGBoost', linewidth=2, markersize=8)
    ax1.plot(years, lgb_auc, '^-', label='LightGBM', linewidth=2, markersize=8)
    ax1.set_xlabel('预测年份', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('各模型AUC趋势', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Recall趋势
    rf_recall = [0.4062, 0.8393, 0.6618, 0.5476]
    xgb_recall = [0.4531, 0.6607, 0.5000, 0.3571]
    lgb_recall = [0.2812, 0.3929, 0.2353, 0.2381]
    
    ax2.plot(years, rf_recall, 'o-', label='Random Forest', linewidth=2, markersize=8)
    ax2.plot(years, xgb_recall, 's-', label='XGBoost', linewidth=2, markersize=8)
    ax2.plot(years, lgb_recall, '^-', label='LightGBM', linewidth=2, markersize=8)
    ax2.set_xlabel('预测年份', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('各模型Recall趋势', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# ==================== 页面3: 消融实验 ====================
elif page == "🔬 消融实验":
    st.markdown('<h1 class="main-header">消融实验分析</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<h2 class="sub-header">特征组合性能对比</h2>', unsafe_allow_html=True)
    st.markdown("验证各特征组合对预测性能的贡献:")
    
    ablation_data = {
        '特征组合': ['仅财务指标', '仅预告数值', '仅BERT语义', '财务+预告', '财务+BERT', '预告+BERT', '全部特征'],
        '维度': [60, 24, 35, 84, 95, 59, 121],
        'AUC均值': [0.6024, 0.7562, 0.5346, 0.7169, 0.6411, 0.7251, 0.7268],
        'AUC标准差': [0.0428, 0.0098, 0.0378, 0.0461, 0.0355, 0.0385, 0.0657],
        'Recall均值': [0.2723, 0.8229, 0.1392, 0.6776, 0.2373, 0.7534, 0.5754],
        'Recall标准差': [0.0869, 0.0882, 0.0346, 0.0625, 0.0911, 0.0921, 0.1314]
    }
    ablation_df = pd.DataFrame(ablation_data)
    st.dataframe(ablation_df.style.format({
        'AUC均值': '{:.4f}',
        'AUC标准差': '{:.4f}',
        'Recall均值': '{:.4f}',
        'Recall标准差': '{:.4f}'
    }), use_container_width=True, hide_index=True)
    
    # 关键发现
    st.markdown('<h2 class="sub-header">关键发现</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.info("🎯 **预告数值特征贡献最大**\n\nAUC=0.7562±0.0098\nRecall=0.8229±0.0882\n\n说明业绩预告数值是最重要的预测信号")
    with col2:
        st.success("💡 **BERT语义特征提升区分能力**\n\n全部特征AUC=0.7268\n财务+预告AUC=0.7169\n\nBERT提升了模型的整体性能")
    
    # 可视化
    st.markdown('<h2 class="sub-header">消融实验可视化</h2>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(ablation_data['特征组合']))
    ax.barh(x, ablation_data['AUC均值'], xerr=ablation_data['AUC标准差'], 
            color='steelblue', alpha=0.8, capsize=5)
    ax.set_yticks(x)
    ax.set_yticklabels(ablation_data['特征组合'])
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('各特征组合AUC对比(含标准差)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig)

# ==================== 页面4: 预测示例 ====================
elif page == "🎯 预测示例":
    st.markdown('<h1 class="main-header">预测示例展示</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<h2 class="sub-header">2025年高风险公司预测</h2>', unsafe_allow_html=True)
    
    example_data = {
        '股票代码': ['600519', '000858', '002594', '600036', '601318', '000001', '600000', '000002'],
        '公司名称': ['贵州茅台', '五粮液', '比亚迪', '招商银行', '中国平安', '平安银行', '浦发银行', '万科A'],
        '预告类型': ['大增', '略增', '扭亏', '续盈', '略降', '续亏', '略增', '大降'],
        '变脸概率': [0.85, 0.72, 0.68, 0.45, 0.38, 0.62, 0.51, 0.78],
        '风险等级': ['极高风险', '高风险', '高风险', '中风险', '低风险', '高风险', '中风险', '高风险']
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True, hide_index=True)
    
    # 风险分布
    st.markdown('<h2 class="sub-header">风险等级分布</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 高风险", "5家", "变脸概率≥60%")
    with col2:
        st.metric("🟡 中风险", "2家", "40%≤概率<60%")
    with col3:
        st.metric("🟢 低风险", "1家", "概率<40%")
    
    # 预警阈值分析
    st.markdown('<h2 class="sub-header">预警阈值分析</h2>', unsafe_allow_html=True)
    
    threshold_data = {
        '阈值': [0.3, 0.4, 0.5, 0.6, 0.7],
        'Precision': [0.0273, 0.0344, 0.0507, 0.0744, 0.1538],
        'Recall': [0.7143, 0.6190, 0.5476, 0.3810, 0.1905],
        'F1': [0.0525, 0.0652, 0.0927, 0.1245, 0.1702],
        '覆盖公司数': [1100, 755, 454, 215, 52]
    }
    threshold_df = pd.DataFrame(threshold_data)
    st.dataframe(threshold_df.style.format({
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1': '{:.4f}'
    }), use_container_width=True, hide_index=True)
    
    st.info("💡 **推荐阈值**: 0.5-0.6之间,在Precision和Recall间取得较好权衡")

# ==================== 页面5: 技术文档 ====================
elif page == "📖 技术文档":
    st.markdown('<h1 class="main-header">技术文档</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 核心技术
    st.markdown('<h2 class="sub-header">核心技术</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 特征工程
        **1. 财务指标 (60维)**
        - 盈利能力: ROE、ROA、净利润率等
        - 偿债能力: 资产负债率、流动比率等
        - 运营能力: 存货周转率、应收账款周转率等
        - 成长能力: 营收增长率、利润增长率等
        
        **2. 预告数值 (24维)**
        - 预告类型编码
        - 预告数值特征
        - 预告修正特征
        - 预告准确度特征
        
        **3. BERT语义 (35维)**
        - FinBERT-tone-chinese模型
        - 文本语义向量
        - 情感极性特征
        - PCA降维至35维
        """)
    
    with col2:
        st.markdown("""
        ### 📊 模型架构
        **1. 滚动预测框架**
        - 3年训练窗口
        - 严格避免未来信息泄露
        - 4轮滚动预测(2022-2025)
        
        **2. 模型集成**
        - 随机森林(最优)
        - XGBoost
        - LightGBM
        
        **3. 类别不平衡处理**
        - BorderlineSMOTE
        - 采样策略: 0.5
        
        **4. 可解释性**
        - SHAP特征重要性
        - Top 20特征分析
        """)
    
    # 技术路线
    st.markdown('<h2 class="sub-header">技术路线图</h2>', unsafe_allow_html=True)
    st.markdown("""
    ```
    数据预处理 → 特征工程 → 模型训练 → 滚动预测 → 性能评估 → 可解释性分析
         ↓            ↓           ↓           ↓           ↓            ↓
    标签生成      多模态融合   SMOTE采样   4轮滚动    消融实验     SHAP分析
    ```
    """)

# ==================== 页面6: 关于系统 ====================
elif page == "ℹ️ 关于系统":
    st.markdown('<h1 class="main-header">关于系统</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📊 系统信息
        - **系统名称**: 业绩变脸早期预警系统
        - **版本**: v1.0
        - **开发时间**: 2026年4月
        - **技术栈**: Python, Streamlit, Scikit-learn
        
        ### 🎯 应用场景
        - 投资决策辅助
        - 风险预警监控
        - 学术研究分析
        - 金融监管参考
        """)
    
    with col2:
        st.markdown("""
        ### 📈 性能指标
        - **最优AUC**: 0.7336
        - **最优Recall**: 0.6137
        - **数据规模**: 26,998样本
        - **特征维度**: 121维
        
        ### 💡 核心创新
        - 多模态特征融合
        - 滚动预测框架
        - 消融实验验证
        - SHAP可解释性
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>业绩变脸早期预警系统 | 基于文本分析与机器学习 | 2026</p>
        <p>本系统仅供学术研究和演示使用</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 页面: 股票预警查询 ====================
elif page == "🔍 股票预警查询":
    st.markdown('<h1 class="main-header">股票预警查询</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<h2 class="sub-header">查询股票历史预告记录</h2>', unsafe_allow_html=True)
    
    # 输入股票代码
    col1, col2 = st.columns([1, 3])
    with col1:
        stock_code = st.text_input("输入股票代码(6位数字)", value="", placeholder="如: 600519")
    
    if stock_code:
        try:
            stock_int = int(stock_code)
        except ValueError:
            st.error("请输入有效的6位数字股票代码")
            st.stop()
        
        # 查询股票数据
        stock_data = sample_df[sample_df['StockCode'] == stock_code]
        
        if len(stock_data) > 0:
            stock_name = stock_data['StockName'].iloc[0]
            st.subheader(f"📊 {stock_name}({stock_code})历史预告记录")
            
            # 显示历史记录
            display_df = stock_data[['Year', 'ForecFinReportType', 'face_change', 'probability']].copy()
            display_df['变脸'] = display_df['face_change'].map({1: '是', 0: '否', -1: '未知'})
            display_df['变脸概率'] = display_df['probability'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "无")
            display_df = display_df.drop(columns=['face_change', 'probability'])
            display_df = display_df.rename(columns={
                'Year': '年份',
                'ForecFinReportType': '预告类型'
            })
            
            st.dataframe(display_df.sort_values('年份', ascending=False), 
                        use_container_width=True, hide_index=True)
            
            # 显示预警结果(2022-2025)
            recent_data = stock_data[stock_data['Year'] >= 2022]
            if len(recent_data) > 0 and recent_data['probability'].notna().any():
                st.subheader("📋 各年份预警结果(2022-2025)")
                
                result_data = []
                for _, row in recent_data.iterrows():
                    if pd.notna(row['probability']):
                        prob = row['probability']
                        if prob >= 0.7:
                            risk = "🔴 极高风险"
                        elif prob >= 0.5:
                            risk = "🟠 高风险"
                        elif prob >= 0.3:
                            risk = "🟡 中风险"
                        else:
                            risk = "🟢 低风险"
                        
                        result_data.append({
                            '年份': row['Year'],
                            '预告类型': row['ForecFinReportType'],
                            '变脸概率': f"{prob:.2%}",
                            '风险等级': risk
                        })
                
                if result_data:
                    result_df = pd.DataFrame(result_data)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                else:
                    st.info("该股票在2022-2025年无预测结果")
            else:
                st.info("该股票在2022-2025年无预告数据,无法预测")
        else:
            st.warning(f"未找到股票代码 {stock_code} 的数据")
            # 显示示例股票代码
            sample_codes = sample_df['StockCode'].unique()[:10]
            st.info(f"可尝试的股票代码: {', '.join(map(str, sample_codes))}")
    else:
        st.info("请输入6位数字股票代码进行查询")
        # 显示示例
        st.markdown("### 示例股票代码:")
        sample_stocks = sample_df[['StockCode', 'StockName']].drop_duplicates().head(10)
        st.dataframe(sample_stocks, use_container_width=True, hide_index=True)

# ==================== 页面: 2025年预测结果 ====================
elif page == "🔮 2025年预测结果":
    st.markdown('<h1 class="main-header">2025年预测结果</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 获取2025年数据
    data_2025 = sample_df[sample_df['Year'] == 2025].copy()
    
    # 统计信息
    st.markdown('<h2 class="sub-header">预测统计</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    probs = data_2025['probability'].values
    with col1:
        st.metric("📊 2025年样本数", f"{len(data_2025)}")
    with col2:
        st.metric("📈 平均变脸概率", f"{probs.mean():.2%}")
    with col3:
        st.metric("🔴 高风险(≥60%)", f"{(probs >= 0.6).sum()}家")
    with col4:
        st.metric("🟢 低风险(<30%)", f"{(probs < 0.3).sum()}家")
    
    # 概率分布
    st.markdown('<h2 class="sub-header">变脸概率分布</h2>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(probs, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=0.3, color='yellow', linestyle='--', label='中风险阈值(0.3)')
    ax.axvline(x=0.6, color='red', linestyle='--', label='高风险阈值(0.6)')
    ax.set_xlabel('变脸概率', fontsize=12)
    ax.set_ylabel('公司数', fontsize=12)
    ax.set_title('2025年业绩预告变脸概率分布', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # 概率筛选功能
    st.markdown('<h2 class="sub-header">变脸概率筛选</h2>', unsafe_allow_html=True)
    st.markdown("输入概率阈值,筛选变脸概率不低于该值的公司:")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        threshold_pct = st.slider("概率阈值(%)", 0, 100, 40, 5)
    
    threshold = threshold_pct / 100.0
    filtered = data_2025[data_2025['probability'] >= threshold].sort_values('probability', ascending=False)
    
    st.info(f"变脸概率 ≥ {threshold_pct}% 的公司共 **{len(filtered)}** 家")
    
    if len(filtered) > 0:
        display_filtered = filtered[['StockCode', 'StockName', 'ForecFinReportType', 'probability']].copy()
        display_filtered['变脸概率'] = display_filtered['probability'].apply(lambda x: f"{x:.2%}")
        display_filtered = display_filtered.drop(columns=['probability'])
        display_filtered = display_filtered.rename(columns={
            'StockCode': '股票代码',
            'StockName': '公司名称',
            'ForecFinReportType': '预告类型'
        })
        st.dataframe(display_filtered.reset_index(drop=True), use_container_width=True, hide_index=True)
    
    # Top 20高风险公司
    st.markdown('<h2 class="sub-header">Top 20 高风险公司</h2>', unsafe_allow_html=True)
    top_20 = data_2025.nlargest(20, 'probability')[['StockCode', 'StockName', 'ForecFinReportType', 'probability']]
    top_20['变脸概率'] = top_20['probability'].apply(lambda x: f"{x:.2%}")
    top_20 = top_20.drop(columns=['probability'])
    top_20 = top_20.rename(columns={
        'StockCode': '股票代码',
        'StockName': '公司名称',
        'ForecFinReportType': '预告类型'
    })
    st.dataframe(top_20.reset_index(drop=True), use_container_width=True, hide_index=True)
