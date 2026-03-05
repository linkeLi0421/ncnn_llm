#include "prompt.h"
#include <sstream>
#include <algorithm>
#include <iostream>

static std::string lstrip_newlines(const std::string& s) {
    size_t start = s.find_first_not_of('\n');
    return (start == std::string::npos) ? "" : s.substr(start);
}

static std::string rstrip_newlines(const std::string& s) {
    size_t end = s.find_last_not_of('\n');
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

// ==========================================
// CHATML TEMPLATE (Qwen3 / MiniCPM4)
// ==========================================

static std::string apply_chatml_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt,
    bool enable_thinking
) {
    std::stringstream prompt;
    bool has_tools = !tools.empty();

    // System message handling
    if (has_tools) {
        prompt << "<|im_start|>system\n";
        if (!messages.empty() && messages[0].role == "system") {
            prompt << messages[0].content << "\n\n";
        }
        prompt << "# Tools\n\n"
               << "You may call one or more functions to assist with the user query.\n\n"
               << "You are provided with function signatures within <tools></tools> XML tags:\n"
               << "<tools>";
        for (const auto& tool : tools) {
            prompt << "\n" << tool.dump();
        }
        prompt << "\n</tools>\n\n"
               << "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
               << "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
    } else {
        if (!messages.empty() && messages[0].role == "system") {
            prompt << "<|im_start|>system\n" << messages[0].content << "<|im_end|>\n";
        }
    }

    // Determine multi-step tool sequence
    bool multi_step_tool = true;
    int last_query_index = (int)messages.size() - 1;

    for (int i = (int)messages.size() - 1; i >= 0; --i) {
        const auto& msg = messages[i];
        bool is_tool_response = false;
        if (msg.content.size() >= 15) {
            if (msg.content.rfind("<tool_response>", 0) == 0 &&
                msg.content.find("</tool_response>") != std::string::npos) {
                is_tool_response = true;
            }
        }
        if (multi_step_tool && msg.role == "user" && !is_tool_response) {
            multi_step_tool = false;
            last_query_index = i;
        }
    }

    // Process messages
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        std::string content = msg.content;

        if (msg.role == "system" && i == 0) continue;

        if (msg.role == "user" || msg.role == "system") {
            prompt << "<|im_start|>" << msg.role << "\n" << content << "<|im_end|>\n";
        }
        else if (msg.role == "assistant") {
            std::string reasoning_content = msg.reasoning_content;
            std::string final_content = content;

            if (reasoning_content.empty() && final_content.find("</think>") != std::string::npos) {
                size_t start_think = final_content.find("<think>");
                size_t end_think = final_content.find("</think>");
                if (start_think != std::string::npos && end_think != std::string::npos) {
                    std::string extracted = final_content.substr(start_think + 7, end_think - (start_think + 7));
                    reasoning_content = lstrip_newlines(rstrip_newlines(extracted));
                    std::string remainder = final_content.substr(end_think + 8);
                    final_content = lstrip_newlines(remainder);
                }
            }

            bool is_after_last_query = ((int)i > last_query_index);
            bool is_last_message = (i == messages.size() - 1);
            bool has_reasoning = !reasoning_content.empty();

            bool show_thinking = enable_thinking && is_after_last_query && (is_last_message || has_reasoning);

            prompt << "<|im_start|>" << msg.role << "\n";

            if (show_thinking && has_reasoning) {
                prompt << "<think>\n" << reasoning_content << "\n</think>\n\n";
            }

            if (!final_content.empty()) {
                prompt << final_content;
            }

            if (!msg.tool_calls.empty()) {
                if (show_thinking || !final_content.empty()) prompt << "\n";
                for (size_t t = 0; t < msg.tool_calls.size(); ++t) {
                    if (t > 0) prompt << "\n";
                    json tc_obj = msg.tool_calls[t];
                    if (tc_obj.contains("function")) tc_obj = tc_obj["function"];

                    prompt << "<tool_call>\n"
                           << "{\"name\": \"" << tc_obj["name"].get<std::string>() << "\", "
                           << "\"arguments\": " << tc_obj["arguments"].dump() << "}\n"
                           << "</tool_call>";
                }
            }
            prompt << "<|im_end|>\n";
        }
        else if (msg.role == "tool") {
            if (i == 0 || messages[i-1].role != "tool") prompt << "<|im_start|>user";
            prompt << "\n<tool_response>\n" << content << "\n</tool_response>";
            if (i == messages.size() - 1 || messages[i+1].role != "tool") prompt << "<|im_end|>\n";
        }
    }

    if (add_generation_prompt) {
        prompt << "<|im_start|>assistant\n";
    }

    return prompt.str();
}

// ==========================================
// YOUTU LLM TEMPLATE
// ==========================================

static std::string apply_youtu_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt
) {
    std::stringstream prompt;
    
    // Build system prompt
    std::string system_prompt;
    bool is_first_sp = true;
    
    // Collect system messages
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            if (is_first_sp) {
                system_prompt += msg.content;
                is_first_sp = false;
            } else {
                system_prompt += "\n\n" + msg.content;
            }
        }
    }
    
    // Check if there are tool messages
    bool has_tool_message = false;
    size_t first_tool_index = messages.size();
    
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        if (!has_tool_message && 
            (msg.role == "tool" || 
             (msg.role == "user" && msg.content.rfind("<tool_response>", 0) == 0 &&
              msg.content.find("</tool_response>") != std::string::npos))) {
            has_tool_message = true;
            first_tool_index = i;
        }
    }
    
    // Add tool descriptions if tools are provided
    if (!tools.empty()) {
        std::string tool_desc = "<|begin_of_tool_description|>Tool calling capabilities.\n"
                               "You may call one or more functions to assist with the user query. You have the following functions available:";
        
        for (const auto& tool : tools) {
            tool_desc += "\n```json\n" + tool.dump() + "\n```";
        }
        
        tool_desc += "\nFor tool call returns, you MUST use the following format:\n"
                    "<tool_call>{\"name\": \"function-name\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}</tool_call>\n"
                    "<|end_of_tool_description|>";
        
        if (system_prompt.empty()) {
            system_prompt = tool_desc;
        } else {
            system_prompt += "\n\n" + tool_desc;
        }
    }
    
    // Output: bos_token + system_prompt
    // Note: bos_token is typically handled by tokenizer, we just output system_prompt
    if (!system_prompt.empty()) {
        prompt << system_prompt;
    }
    
    // Process messages
    bool is_first = false;  // Not used in YouTu template like ChatML
    bool is_tool = false;
    bool is_last_user = false;
    
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        
        if (msg.role == "system") continue;  // Already handled above
        
        if (msg.role == "user") {
            is_tool = false;
            is_first = false;
            is_last_user = true;
            
            // Check if this is a tool response message
            if (msg.content.rfind("<tool_response>", 0) == 0 &&
                msg.content.find("</tool_response>") != std::string::npos) {
                // Tool response is already formatted correctly
                prompt << "<|User|>" << msg.content;
            } else {
                prompt << "<|User|>" << msg.content;
            }
        }
        else if (msg.role == "assistant") {
            is_last_user = false;
            std::string content = msg.content;
            
            // Handle <think> tags if present
            std::string think_content;
            size_t think_start = content.find("<think>");
            size_t think_end = content.find("</think>");
            
            if (think_start != std::string::npos && think_end != std::string::npos) {
                think_content = content.substr(think_start + 7, think_end - (think_start + 7));
                think_content = lstrip_newlines(rstrip_newlines(think_content));
                content = content.substr(0, think_start) + content.substr(think_end + 8);
                content = lstrip_newlines(content);
            }
            
            prompt << "<|Assistant|>";
            
            // Output think content if present
            if (!think_content.empty()) {
                prompt << "<think>" << think_content << "</think>";
            }
            
            // Output main content
            if (!content.empty()) {
                prompt << content;
            }
            
            // Output tool calls
            if (!msg.tool_calls.empty()) {
                for (const auto& tc : msg.tool_calls) {
                    json tc_obj = tc;
                    if (tc_obj.contains("function")) tc_obj = tc_obj["function"];
                    
                    prompt << "<tool_call>{\"name\": \"" 
                           << tc_obj["name"].get<std::string>() << "\", "
                           << "\"arguments\": " << tc_obj["arguments"].dump() 
                           << "}</tool_call>";
                }
            }
        }
        else if (msg.role == "tool") {
            // Tool messages are appended to the next user message or output directly
            if (i == 0 || messages[i-1].role != "tool") {
                prompt << "<|User|><tool_response>" << msg.content << "</tool_response>";
            } else {
                // Multiple tool messages in sequence
                prompt << msg.content;
            }
        }
    }
    
    if (add_generation_prompt) {
        prompt << "<|Assistant|>";
    }
    
    return prompt.str();
}

// ==========================================
// PUBLIC API
// ==========================================

std::string apply_chat_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt,
    bool enable_thinking
) {
    return apply_chatml_template(messages, tools, add_generation_prompt, enable_thinking);
}

std::string apply_youtu_chat_template(
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt
) {
    return apply_youtu_template(messages, tools, add_generation_prompt);
}

std::string apply_chat_template(
    TemplateType type,
    const std::vector<Message>& messages,
    const std::vector<json>& tools,
    bool add_generation_prompt,
    bool enable_thinking
) {
    switch (type) {
        case TemplateType::YOUTU:
            return apply_youtu_template(messages, tools, add_generation_prompt);
        case TemplateType::CHATML:
        default:
            return apply_chatml_template(messages, tools, add_generation_prompt, enable_thinking);
    }
}
