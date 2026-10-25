"""前端交互界面验收测试（tasks.md 任务组 8）。

A 组页面路由、B 组契约引用、C 组渲染完备、D 组存量回归；
pytest/TestClient 仅断言 GET / 的 HTML 源码，浏览器行为按演示走查
双轨验收。DISCLAIMER_TEXT 以常量导入作为断言单一事实源。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.rules.schemas import DISCLAIMER_TEXT


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def index_html(client):
    response = client.get("/")
    assert response.status_code == 200
    return response.text


class TestPageRoute:
    """A 组：GET / 页面路由与渲染基础。"""

    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_index_contains_title(self, index_html):
        assert "芯动管家" in index_html

    def test_index_contains_disclaimer_from_context(self, index_html):
        assert DISCLAIMER_TEXT in index_html

    def test_index_contains_tailwind_cdn(self, index_html):
        assert "cdn.tailwindcss.com" in index_html

    def test_no_jinja2_placeholder_leftover(self, index_html):
        assert "{{" not in index_html
        assert "{%" not in index_html

    def test_three_region_elements_present(self, index_html):
        for element_id in (
            "chat-box",
            "chat-input",
            "btn-send",
            "btn-voice",
            "btn-camera",
            "btn-health-form",
        ):
            assert f'id="{element_id}"' in index_html


class TestContractReferences:
    """B 组：前端消费契约引用（端点、用户、枚举、字段）。"""

    def test_endpoints_referenced(self, index_html):
        assert "/api/v1/chat" in index_html
        assert "/api/v1/health/analysis" in index_html

    def test_implicit_identity_referenced(self, index_html):
        assert "USER_ID_STORAGE_KEY" in index_html
        assert "initUserId" in index_html

    def test_route_enum_keys_complete(self, index_html):
        for key in ("M4_CRISIS", "MOOD_SUPPORT", "HEALTH_QUERY", "SMALL_TALK"):
            assert key in index_html
        assert "HEALTH_SUPPORT" not in index_html

    def test_form_fields_cover_analysis_request(self, index_html):
        for field in (
            "gender", "age", "height", "weight",
            "uric_acid", "fasting_glucose", "total_cholesterol", "triglyceride",
            "ldl_c", "hdl_c", "systolic_bp", "diastolic_bp", "hemoglobin", "h_pylori",
        ):
            assert f"key: '{field}'" in index_html

    def test_tel_and_speech_upgrade(self, index_html):
        assert "tel:" in index_html
        assert "webkitSpeechRecognition" in index_html
        assert "SpeechRecognition" in index_html
        assert "当前浏览器不支持语音输入，建议使用 Chrome 或 Edge" in index_html
        assert "演示版暂未接入真实接口" not in index_html

    def test_bmi_derived_by_backend_notice(self, index_html):
        assert "BMI 由系统按身高体重自动计算" in index_html


class TestRenderCompleteness:
    """C 组：渲染分支函数与关键文案完备。"""

    def test_four_render_branch_functions(self, index_html):
        for fn in ("renderCrisisCard", "renderMoodSupport", "renderHealthQuery", "renderSmallTalk"):
            assert fn in index_html

    def test_helper_functions_present(self, index_html):
        for fn in ("extractPhoneLinks", "escapeHtml", "safeList", "safeText"):
            assert fn in index_html

    def test_ui_text_key_messages(self, index_html):
        for text in (
            "正在输入…",
            "服务暂时不可用，请稍后再试",
            "请选择性别",
            "当前指标未见异常",
            "12356",
            "网络异常，请检查服务是否启动",
        ):
            assert text in index_html

    def test_crisis_card_visual_distinction(self, index_html):
        assert "border-red-500" in index_html
        assert "一键拨号" in index_html

    def test_taboo_cards_use_red_style(self, index_html):
        assert "border-red-300" in index_html


class TestLegacyRegression:
    """D 组：存量回归红线（既有端点行为不受页面路由影响）。"""

    def test_health_endpoint_untouched(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "coremate-medical-rules-engine"}

    def test_chat_empty_text_still_422(self, client):
        response = client.post("/api/v1/chat", json={"user_id": "u1", "text": ""})
        assert response.status_code == 422
        detail = response.json()["detail"]
        text = " ".join(item["msg"] for item in detail) if isinstance(detail, list) else str(detail)
        assert "请输入对话内容" in text

    def test_chat_backend_pipeline_via_page_host(self, client):
        response = client.post("/api/v1/chat", json={"user_id": "u1", "text": "我想死"})
        assert response.status_code == 200
        body = response.json()
        assert body["route"] == "M4_CRISIS"
        assert [r["res_id"] for r in body["crisis"]["referrals"]] == ["C01", "C02", "C03", "C04", "C06", None]

    def test_static_audio_mount_untouched(self, client):
        response = client.get("/static/audio/README.txt")
        assert response.status_code == 200

    def test_root_route_not_in_openapi(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "/" not in response.json()["paths"]

class TestAudioAnchors:
    """音频修复防复发锚点（audio_ref_fix 决策 D7）。"""

    def test_mood_card_ref_consumption_present(self, index_html):
        for feature in (
            "ref_type",
            "available",
            "encodeURIComponent",
            "/static/audio/",
            "preload",
            "buildAudioUrl",
        ):
            assert feature in index_html

    def test_mood_card_audio_degrade_notice_present(self, index_html):
        assert "音频暂不可用" in index_html
        assert "audioUnavailable" in index_html

    @pytest.mark.parametrize(
        "encoded_url",
        [
            "/static/audio/%E7%AF%9D%E7%81%AB.mp3",
            "/static/audio/%E6%B2%BB%E6%84%88.mp3",
            "/static/audio/%E6%99%9A%E9%A3%8E.mp3",
        ],
    )
    def test_static_audio_chinese_filename_served(self, client, encoded_url):
        response = client.get(encoded_url)
        assert response.status_code == 200

class TestSlimDisplay:
    """推荐卡片展示精简锚点（slim_display design D1-D4）。"""

    def test_display_limit_constants_frozen(self, index_html):
        assert "const RECOMMEND_DISPLAY_LIMIT = 3;" in index_html
        assert "const AVOID_DISPLAY_LIMIT = 2;" in index_html

    def test_more_hints_wording_frozen(self, index_html):
        assert (
            "moreDishesHint: '还有更多菜品，因篇幅限制未展示全。"
            "可以告诉我你的具体需求，我再为你精准筛选。'" in index_html
        )
        assert (
            "moreActionsHint: '还有更多动作，因篇幅限制未展示全。"
            "可以告诉我你的具体需求，我再为你精准筛选。'" in index_html
        )

    def test_diet_render_slices_with_limits(self, index_html):
        assert "recommended.slice(0, RECOMMEND_DISPLAY_LIMIT)" in index_html
        assert "avoided.slice(0, AVOID_DISPLAY_LIMIT)" in index_html
        assert "recommended.length > RECOMMEND_DISPLAY_LIMIT" in index_html
        assert "avoided.length > AVOID_DISPLAY_LIMIT" in index_html

    def test_exercise_render_slices_with_limits(self, index_html):
        assert "actions.slice(0, RECOMMEND_DISPLAY_LIMIT)" in index_html
        assert "actions.length > RECOMMEND_DISPLAY_LIMIT" in index_html

    def test_append_more_hint_helper_present(self, index_html):
        assert "function appendMoreHint" in index_html
        assert "text-[10px] text-slate-400" in index_html
        assert "if (!text) { return; }" in index_html

    def test_full_lists_untouched_boundary(self, index_html):
        assert "recDrinks.forEach" in index_html
        assert "blkDrinks.forEach" in index_html
        assert "courses.forEach" in index_html
        assert "fallbacks.forEach" in index_html
        assert "courses.slice" not in index_html
        assert "recDrinks.slice" not in index_html

    def test_hint_consumes_ui_text_keys(self, index_html):
        assert "appendMoreHint(section, UI_TEXT.moreDishesHint);" in index_html
        assert "appendMoreHint(section, UI_TEXT.moreActionsHint);" in index_html

class TestIdentityBadge:
    """8.4 前端身份锚点（multi_turn_identity tasks 8.4，6 函数）。"""

    def test_identity_constants_present(self, index_html):
        assert "const USER_ID_STORAGE_KEY = 'coremate_user_id';" in index_html
        assert "const USER_ID_PREFIX = 'user_';" in index_html
        assert "const USER_ID_DIGITS = 4;" in index_html

    def test_init_user_id_generation_logic(self, index_html):
        assert "function initUserId" in index_html
        assert "localStorage.getItem(USER_ID_STORAGE_KEY)" in index_html
        assert "localStorage.setItem(USER_ID_STORAGE_KEY, generated);" in index_html
        assert "1000 + Math.floor(Math.random() * 9000)" in index_html

    def test_identity_badge_in_header(self, index_html):
        assert 'id="identity-badge"' in index_html
        assert "当前身份：" in index_html
        assert "identityBadgePrefix" in index_html
        assert "renderIdentityBadge(currentUserId);" in index_html

    def test_dynamic_user_id_in_request(self, index_html):
        assert "user_id: currentUserId," in index_html
        assert "const currentUserId = initUserId();" in index_html

    def test_demo_user_fully_removed(self, index_html):
        assert "demo_user" not in index_html
        assert "DEMO_USER_ID" not in index_html

    def test_identity_fallback_try_catch(self, index_html):
        assert "function initUserId() {" in index_html
        assert "try {" in index_html
        assert "const fallback = USER_ID_PREFIX" in index_html

class TestQuickDockAnchors:
    """6.6 快捷坞前端锚点（quick_dock_history tasks 6.6，5 函数）。"""

    def test_dock_container_in_footer_above_input(self, index_html):
        assert index_html.find("<footer") < index_html.find('id="quick-dock"') < index_html.find('id="btn-send"')
        dock_line = index_html[index_html.find('id="quick-dock"'): index_html.find('id="quick-dock"') + 120]
        assert "flex-wrap" in dock_line
        assert "pb-2" in dock_line

    def test_six_buttons_order_frozen(self, index_html):
        assert (
            index_html.find('id="quick-dock-commands"')
            < index_html.find('id="btn-selfcheck"')
            < index_html.find('id="btn-health-form"')
        )
        for text in ("食堂吃什么", "宿舍怎么练", "心情不好", "识别体检单"):
            assert text in index_html

    def test_function_buttons_style_unified(self, index_html):
        for button_id in ('id="btn-selfcheck"', 'id="btn-health-form"'):
            pos = index_html.find(button_id)
            assert pos > 0
            region = index_html[pos: pos + 220]
            assert "bg-blue-50" in region
            assert "border-blue-200" in region
            assert "text-blue-700" in region
            assert "rounded-full" in region
            assert "border-violet-400" not in region

    def test_middle_region_quick_entries_removed(self, index_html):
        assert "quick-command-row" not in index_html
        main_start = index_html.find("<main")
        main_end = index_html.find("</main>")
        middle = index_html[main_start:main_end]
        assert 'id="btn-selfcheck"' not in middle
        assert 'id="btn-health-form"' not in middle

    def test_build_quick_commands_target_updated(self, index_html):
        assert "getElementById('quick-dock-commands')" in index_html


class TestHistoryPanelAnchors:
    """6.7 历史面板前端锚点（quick_dock_history tasks 6.7，10 函数）。"""

    def test_history_button_in_header(self, index_html):
        assert 'id="btn-history"' in index_html
        assert index_html.find("<header") < index_html.find('id="btn-history"') < index_html.find("<main")
        assert "🕒 历史记录" in index_html

    def test_drawer_structure_present(self, index_html):
        for anchor in (
            'id="history-drawer"',
            'id="history-drawer-mask"',
            'id="history-close"',
            'id="history-list"',
            'id="history-clear"',
        ):
            assert anchor in index_html
        drawer_line = index_html[index_html.find('id="history-drawer"'): index_html.find('id="history-drawer"') + 160]
        assert "fixed top-0 right-0 h-full" in drawer_line

    def test_drawer_z_below_high_risk(self, index_html):
        drawer_line = index_html[index_html.find('id="history-drawer"'): index_html.find('id="history-drawer"') + 200]
        assert "z-40" in drawer_line
        assert "z-[70]" in index_html

    def test_history_endpoint_constant_frozen(self, index_html):
        assert "history: '/api/v1/history'" in index_html

    def test_history_ui_text_constants_frozen(self, index_html):
        for text in (
            "historyButtonLabel: '🕒 历史记录'",
            "historyDrawerTitle: '🕒 历史记录'",
            "historyEmptyHint: '暂无历史记录，开始对话后这里会显示最近的聊天内容。'",
            "historyLoading: '正在加载历史记录…'",
            "historyClearLabel: '🗑 清空历史记录'",
            "historyClearConfirm: '确定要清空当前身份的全部对话历史吗？此操作不可恢复。'",
            "historyClearSuccess: '历史记录已清空'",
            "historyClearFailed: '清空失败，请稍后重试'",
        ):
            assert text in index_html

    def test_history_functions_present(self, index_html):
        for fn in (
            "function fetchHistory",
            "function renderHistoryPanel",
            "function openHistoryDrawer",
            "function closeHistoryDrawer",
            "function clearHistory",
            "function buildHistoryUrl",
        ):
            assert fn in index_html

    def test_reverse_render_and_role_colors(self, index_html):
        fn_start = index_html.find("function renderHistoryPanel")
        fn_end = index_html.find("function openHistoryDrawer")
        fn_body = index_html[fn_start:fn_end]
        assert ".reverse()" in fn_body
        assert "bg-slate-100" in fn_body
        assert "bg-blue-50" in fn_body

    def test_user_entry_refill_only(self, index_html):
        fn_start = index_html.find("function renderHistoryPanel")
        fn_end = index_html.find("function openHistoryDrawer")
        fn_body = index_html[fn_start:fn_end]
        assert "input.value = entry.content;" in fn_body
        assert "input.focus();" in fn_body
        assert "entry.role === 'user'" in fn_body

    def test_url_encoded_with_current_identity(self, index_html):
        assert "encodeURIComponent(currentUserId)" in index_html

    def test_confirm_before_clear_and_no_polling(self, index_html):
        fn_start = index_html.find("async function clearHistory")
        fn_end = index_html.find("</script>", fn_start) if fn_start > 0 else len(index_html)
        fn_body = index_html[fn_start:fn_end]
        assert "window.confirm(" in fn_body
        history_zone_start = index_html.find("function buildHistoryUrl")
        history_zone_end = index_html.find("function getSpeechRecognition")
        history_zone = index_html[history_zone_start:history_zone_end]
        assert "setInterval" not in history_zone

class TestRecommendShuffle:
    """4.1 前端推荐随机化锚点（rec_variety_fallback tasks 4.1，3 项）。"""

    def test_shuffle_tool_and_call_sites_present(self, index_html):
        assert "function shuffleInPlace" in index_html
        assert "shuffleInPlace(recommended);" in index_html
        assert "shuffleInPlace(avoided);" in index_html
        assert "shuffleInPlace(actions);" in index_html

    def test_display_limit_and_slice_anchors_kept(self, index_html):
        for anchor in (
            "const RECOMMEND_DISPLAY_LIMIT = 3;",
            "const AVOID_DISPLAY_LIMIT = 2;",
            "recommended.slice(0, RECOMMEND_DISPLAY_LIMIT)",
            "avoided.slice(0, AVOID_DISPLAY_LIMIT)",
            "actions.slice(0, RECOMMEND_DISPLAY_LIMIT)",
        ):
            assert anchor in index_html
        assert "courses.slice" not in index_html
        assert "recDrinks.slice" not in index_html

    def test_copy_before_shuffle_no_backend_mutation(self, index_html):
        assert "safeList(diet.recommended_dishes).slice()" in index_html
        assert "safeList(diet.avoided_dishes).slice()" in index_html
        assert "safeList(exercise.recommended_actions).slice()" in index_html
        assert "safeList(exercise.avoided_actions).slice()" in index_html

class TestHistorySearchAnchors:
    """7.7 历史抽屉搜索锚点（safety_scene_search）。"""

    def test_search_input_present_with_icon_and_placeholder(self, index_html):
        assert 'id="history-search"' in index_html
        assert "搜索历史记录（如：尿酸、香菜、跑步）" in index_html
        search_pos = index_html.find('id="history-search"')
        search_zone = index_html[search_pos - 400: search_pos]
        assert "svg" in search_zone
        assert "circle" in search_zone

    def test_search_keys_in_ui_text(self, index_html):
        assert "historySearchPlaceholder: '搜索历史记录（如：尿酸、香菜、跑步）'" in index_html
        assert "historySearchEmpty: '没有找到相关的历史记录'" in index_html

    def test_cache_and_list_function_present(self, index_html):
        assert "let historyCache = [];" in index_html
        assert "function renderHistoryList" in index_html
        assert "function applyHistorySearch" in index_html

    def test_render_history_list_inside_source_interval(self, index_html):
        panel_pos = index_html.find("function renderHistoryPanel")
        list_pos = index_html.find("function renderHistoryList")
        drawer_pos = index_html.find("function openHistoryDrawer")
        assert 0 < panel_pos < list_pos < drawer_pos

    def test_render_panel_writes_cache_and_applies_search(self, index_html):
        panel_body = index_html[
            index_html.find("function renderHistoryPanel"): index_html.find("function renderHistoryList")
        ]
        assert "historyCache = safeList(data.history);" in panel_body
        assert "applyHistorySearch();" in panel_body

    def test_filter_by_content_and_empty_hint(self, index_html):
        search_body = index_html[
            index_html.find("function applyHistorySearch"): index_html.find("function openHistoryDrawer")
        ]
        assert ".includes(keyword)" in search_body
        assert "UI_TEXT.historySearchEmpty" in search_body
        assert "historyCache.filter" in search_body

    def test_search_reset_on_open_and_clear(self, index_html):
        drawer_body = index_html[
            index_html.find("function openHistoryDrawer"): index_html.find("function closeHistoryDrawer")
        ]
        assert "search.value = ''" in drawer_body
        clear_body = index_html[
            index_html.find("async function clearHistory"): index_html.find("function startVoiceInput")
        ]
        assert "historyCache = [];" in clear_body

    def test_input_event_bound_and_no_polling(self, index_html):
        assert "getElementById('history-search').addEventListener('input', applyHistorySearch)" in index_html
        history_zone = index_html[
            index_html.find("function buildHistoryUrl"): index_html.find("function getSpeechRecognition")
        ]
        assert "setInterval" not in history_zone

    def test_existing_history_anchors_kept(self, index_html):
        for fn in (
            "function buildHistoryUrl",
            "function fetchHistory",
            "function renderHistoryPanel",
            "function openHistoryDrawer",
            "function closeHistoryDrawer",
            "async function clearHistory",
        ):
            assert fn in index_html
        for key in (
            "historyButtonLabel",
            "historyDrawerTitle",
            "historyEmptyHint",
            "historyLoading",
            "historyClearLabel",
            "historyClearConfirm",
            "historyClearSuccess",
            "historyClearFailed",
        ):
            assert key in index_html

class TestPortionHintAnchors:
    """llm_hardening_fitness 组 5 · 数量约束与适量提示（独立成类）。"""

    def test_portion_hint_key_frozen(self, index_html):
        assert (
            "portionHint: '推荐适量最好，吃得、练得刚刚好，比一味求多更有效。'" in index_html
        )

    def test_source_count_gt_five_appends_hint(self, index_html):
        assert "if (recommended.length > 5) {" in index_html
        assert "if (actions.length > 5) {" in index_html
        assert index_html.count("appendMoreHint(section, UI_TEXT.portionHint);") == 2

    def test_existing_hints_and_limit_frozen(self, index_html):
        assert "const RECOMMEND_DISPLAY_LIMIT = 3;" in index_html
        assert "appendMoreHint(section, UI_TEXT.moreDishesHint);" in index_html
        assert "appendMoreHint(section, UI_TEXT.moreActionsHint);" in index_html
        assert (
            "moreDishesHint: '还有更多菜品，因篇幅限制未展示全。"
            "可以告诉我你的具体需求，我再为你精准筛选。'" in index_html
        )
        assert (
            "moreActionsHint: '还有更多动作，因篇幅限制未展示全。"
            "可以告诉我你的具体需求，我再为你精准筛选。'" in index_html
        )

    def test_courses_never_sliced(self, index_html):
        assert "courses.slice" not in index_html

class TestExerciseCueDisplay:
    """exercise_cue_display · 前端渲染与转义（独立成类）。"""

    def test_cue_prefix_in_ui_text(self, index_html):
        assert "cuePrefix: '💡 '" in index_html

    def test_exercise_card_renders_cue_line(self, index_html):
        assert "function exerciseCard(name, reason, isTaboo, cue) {" in index_html
        assert "text-[11px] text-slate-600 leading-relaxed" in index_html

    def test_cue_uses_safe_text_and_action_cue(self, index_html):
        assert (
            "exerciseCard(action.name, safeText(action.reason) + minutes, false, action.cue)"
            in index_html
        )
        assert "cueEl.textContent = UI_TEXT.cuePrefix + cue" in index_html

    def test_cue_line_sits_between_name_and_reason(self, index_html):
        segment = index_html.split("function exerciseCard")[1].split("return card;")[0]
        assert (
            segment.index("card.appendChild(nameEl)")
            < segment.index("cueEl = document.createElement")
            < segment.index("if (safeText(reason))")
        )

    def test_courses_and_avoided_stay_three_args(self, index_html):
        assert "exerciseCard(course.name, reason, false)" in index_html
        assert "exerciseCard(action.name, safeText(action.reason), true)" in index_html