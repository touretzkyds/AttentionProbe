"""
Attention visualization module for the AttentionProbe application.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox
from typing import List, Tuple, Optional, Dict, Any

from config import UI_CONFIG, DEMO_CONFIGS
from utils import ModelManager, find_difference


class AttentionVisualizer:
    """Handles attention visualization for T5 models."""
    
    def __init__(self, model_manager: ModelManager, prompts: List[str], demo_type : str):
        """
        Initialize the attention visualizer.
        
        Args:
            model_manager: Model manager instance
            prompts: List of prompts to visualize
        """
        self.model_manager = model_manager
        self.model_manager.load_model()
        self.prompts = prompts
        self.tokenizer = model_manager.tokenizer
        self.config = model_manager.config

        self.cur_layer_idx = 0
        self.cur_head_idx = 0
        self.cur_overall_idx = 0
        self._handle_demo_type(demo_type)
        
        # Visualization state
        self.fig = None
        self.axs = None
        self.tooltips = {}
        self.highlight_indices = None

        # Colorbars
        self.cb1 = None
        self.cb2 = None
        self.cb3 = None

        # Textboxes (to jump to different heads)
        self.layer_textbox_ax = None
        self.head_textbox_ax = None
        self.layer_textbox = None
        self.head_textbox = None

        # Labels for interesting attention heads (only fully init-ed for demos, NOT base)
        self.attention_head_count_label_ax = None

        # Process inputs
        self._process_inputs()

        # Safety checker for textbox inputting
        self.safe_tamper_flag = False
        
        # UI elements
        self.range_slider = None
        self.layer_textbox = None
        self.head_textbox = None
        
        # Visualization elements
        self.im1 = None
        self.im2 = None
        self.im_diff = None
        self.original_range_im1 = None
        self.original_range_im2 = None
        self.original_range_imdiff = None
        self.all_lines1 = []
        self.all_lines2 = []
        self.all_lines3 = []
        self.hovered_lines = []

    def _handle_demo_type(self, demo_type : str):
        if demo_type == 'base': # this is the baseline visualization, with all the attention heads
            self.cur_layer_idx = 0
            self.cur_head_idx = 0
            self.interesting_attns = [] # since this is the baseline, there are no "interesting" heads
        else: # we are doing an actual demo
            self.interesting_attns = DEMO_CONFIGS[demo_type]['interesting_heads']
            self.cur_layer_idx = self.interesting_attns[0][0]
            self.cur_head_idx = self.interesting_attns[0][1]

        
    def _process_inputs(self):
        """Process input prompts and get model outputs."""
        # Get model outputs with attention
        self.outputs = self.model_manager.get_attention_outputs(
            self.prompts, 
            max_length=UI_CONFIG['max_generation_length']
        )
        
        # Tokenize inputs
        inputs = self.tokenizer(
            self.prompts,
            padding=True,#this is where the padding is included; used for sequences so that they all have the same length
            return_tensors="pt",
            return_attention_mask=True
        )
        
        self.inputs_ids = inputs.input_ids
        self.tokens1 = self.tokenizer.convert_ids_to_tokens(self.inputs_ids[0])
        self.tokens2 = self.tokenizer.convert_ids_to_tokens(self.inputs_ids[1])
        
        # Find difference between tokens
        self.diff_idx = find_difference(self.tokens1, self.tokens2)
        self.diff_t1 = self.tokens1[self.diff_idx]
        self.diff_t2 = self.tokens2[self.diff_idx]
        self.tokens3 = [
            self.tokens1[i] if i != self.diff_idx else f"{self.diff_t1}/{self.diff_t2}" 
            for i in range(len(self.tokens1))
        ]
        
        # Get attention data
        self.cur_layer_attentions = self.outputs.encoder_attentions[self.cur_layer_idx]
        self.num_heads_per_layer = self.cur_layer_attentions.shape[1]
        self.total_num_layers = self.config.num_layers
        
        print(f"Tokens 1: {self.tokens1}")
        print(f"Tokens 2: {self.tokens2}")
        
    def _compute_new_range(self, im, percent: float):
        """Compute new range for image normalization."""
        if im == self.im1:
            cur_min, cur_max = self.original_range_im1
        elif im == self.im2:
            cur_min, cur_max = self.original_range_im2
        else:
            cur_min, cur_max = self.original_range_imdiff
            
        if abs(cur_min) < abs(cur_max):
            new_total_range = (cur_max - cur_min) * percent
            new_max = cur_min + new_total_range
            new_min = cur_min
        else:
            new_total_range = (cur_max - cur_min) * percent
            new_min = cur_max - new_total_range
            new_max = cur_max
            
        im.set_clim(new_min, new_max)
        
    def _slider_update(self, val):
        """Update visualization based on slider value."""
        self._compute_new_range(self.im1, val)
        self._compute_new_range(self.im2, val)
        self._compute_new_range(self.im_diff, val)
        self.fig.canvas.draw_idle()
        
    def _init_slider(self):
        """Initialize the range slider."""
        range_slider_ax = self.fig.add_axes([0.1, 0.03, 0.2, 0.03])
        self.range_slider = Slider(
            range_slider_ax,
            'Adjust Range', 
            UI_CONFIG['slider_range'][0], 
            UI_CONFIG['slider_range'][1], 
            valinit=UI_CONFIG['slider_default']
        )
        self.range_slider.poly.set_alpha(0.0)
        self.range_slider.on_changed(self._slider_update)
            
    def _init_tooltip(self):
        """Initialize tooltips for the visualization."""
        for ax in self.axs.flatten():
            annotation = ax.annotate(
                "", xy=(0, 0), xytext=(-60, 10), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->", color="white"),
                zorder=100
            )
            ax.title.set_zorder(1)
            annotation.set_zorder(100)
            annotation.set_visible(False)
            self.tooltips[ax] = annotation
            
    def _text_colorchange(self, ax):
        """Change text color to highlight differences."""
        y_labels = ax.get_yticklabels()
        if self.diff_idx < len(y_labels):
            y_labels[self.diff_idx].set_color(UI_CONFIG['highlight_color'])
            y_labels[self.diff_idx].set_fontweight('bold')
            
        x_labels = ax.get_xticklabels()
        if self.diff_idx < len(x_labels):
            x_labels[self.diff_idx].set_color(UI_CONFIG['highlight_color'])
            x_labels[self.diff_idx].set_fontweight('bold')
            
        ax.figure.canvas.draw()
        
    def _draw_line_diff(self, ax, i, j, attention, spacing):
        """Draw attention lines for difference visualization."""
        y1 = 1 - (i + 0.6) * spacing
        y2 = 1 - (j + 0.6) * spacing
        alpha = abs(attention)
        
        if attention < 0:
            return ax.plot(
                [0, 1], [y1, y2],
                transform=ax.transAxes,
                color='red',
                alpha=alpha,
                linewidth=1
            )[0]
        else:
            return ax.plot(
                [0, 1], [y1, y2],
                color='green',
                transform=ax.transAxes,
                alpha=alpha,
                linewidth=1
            )[0]
            
    def _draw_line_prompts(self, ax, i, j, attention, spacing, color):
        """Draw attention lines for prompt visualization."""
        y1 = 1 - (i + 0.6) * spacing
        y2 = 1 - (j + 0.6) * spacing
        alpha = attention
        
        return ax.plot(
            [0, 1], [y1, y2],
            color=color,
            transform=ax.transAxes,
            alpha=alpha,
            linewidth=1
        )[0]
        
    def _compute_tokenbounds(self, tokens, spacing):
        """Compute token boundaries for visualization."""
        token_bounds = []
        for i in range(len(tokens)):
            y_center = 1 - (i + 0.6) * spacing
            height = spacing * 0.9
            ymin = y_center - height / 2
            ymax = y_center + height / 2
            token_bounds.append((ymin, ymax))
        return token_bounds
        
    def _init_line_visualizations(self, ax4, ax5, ax6):
        """Initialize line-based visualizations."""
        spacing1 = 1 / len(self.tokens1)
        
        # Draw tokens and lines for first prompt
        for i, token in enumerate(self.tokens1):
            y = 1 - (i + 0.6) * spacing1
            color = UI_CONFIG['highlight_color'] if i == self.diff_idx else UI_CONFIG['normal_color']
            weight = 'bold' if i == self.diff_idx else 'normal'
            
            ax4.text(0, y, token, ha='right', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax4.transAxes)
            ax4.text(1, y, token, ha='left', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax4.transAxes)
        for i in range(len(self.tokens1)):
            for j in range(len(self.tokens1)):
                attention_val = self.cur_layer_attentions[0, self.cur_head_idx, i, j].item()
                line = self._draw_line_prompts(ax4, i, j, attention_val, spacing1, 'blue')
                self.all_lines1.append(line)
        ax4.axis("off")
                
        # Similar for second prompt
        spacing2 = 1 / len(self.tokens2)
        for i, token in enumerate(self.tokens2):
            y = 1 - (i + 0.6) * spacing2
            color = UI_CONFIG['highlight_color'] if i == self.diff_idx else UI_CONFIG['normal_color']
            weight = 'bold' if i == self.diff_idx else 'normal'
            
            ax5.text(0, y, token, ha='right', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax5.transAxes)
            ax5.text(1, y, token, ha='left', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax5.transAxes)
        for i in range(len(self.tokens2)):
            for j in range(len(self.tokens2)):
                attention_val = self.cur_layer_attentions[1, self.cur_head_idx, i, j].item()
                line = self._draw_line_prompts(ax5, i, j, attention_val, spacing2, 'blue')
                self.all_lines2.append(line)
        ax5.axis("off")
                
        # Difference visualization
        spacing3 = 1 / len(self.tokens3)
        for i, token in enumerate(self.tokens3):
            y = 1 - (i + 0.6) * spacing3
            color = UI_CONFIG['highlight_color'] if i == self.diff_idx else UI_CONFIG['normal_color']
            weight = 'bold' if i == self.diff_idx else 'normal'
            
            ax6.text(0, y, token, ha='right', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax6.transAxes)
            ax6.text(1, y, token, ha='left', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax6.transAxes)
        for i in range(len(self.tokens3)):
            for j in range(len(self.tokens3)):
                attn1 = self.cur_layer_attentions[0, self.cur_head_idx, i, j].item()
                attn2 = self.cur_layer_attentions[1, self.cur_head_idx, i, j].item()
                diff = attn1 - attn2
                line = self._draw_line_diff(ax6, i, j, diff, spacing3)
                self.all_lines3.append(line)
        ax6.axis("off")
                
    def _init_matrix_visualizations(self, ax1, ax2, ax3):
        """Initialize matrix-based visualizations."""
        # First prompt attention matrix
        attn1 = self.cur_layer_attentions[0, self.cur_head_idx].detach().numpy()
        ax1.set_title("Sentence 1 Attention")
        ax1.set_xticks(np.arange(len(self.tokens1)))
        ax1.set_yticks(np.arange(len(self.tokens1)))
        ax1.set_xticklabels(self.tokens1, rotation=90)
        ax1.set_yticklabels(self.tokens1)
        self.im1 = ax1.imshow(attn1)
        self._text_colorchange(ax1)
        self.cb1 = self.fig.colorbar(self.im1, ax=ax1, shrink=0.8, pad=0.1)
        
        # Second prompt attention matrix
        attn2 = self.cur_layer_attentions[1, self.cur_head_idx].detach().numpy()
        ax2.set_title("Sentence 2 Attention")
        ax2.set_xticks(np.arange(len(self.tokens2)))
        ax2.set_yticks(np.arange(len(self.tokens2)))
        ax2.set_xticklabels(self.tokens2, rotation=90)
        ax2.set_yticklabels(self.tokens2)
        self.im2 = ax2.imshow(attn2)
        self._text_colorchange(ax2)
        self.cb2 = self.fig.colorbar(self.im2, ax=ax2, shrink=0.8, pad=0.1)
        
        # Difference matrix
        diff = attn1 - attn2
        ax3.set_title(f'Difference')
        ax3.set_xticks(np.arange(len(self.tokens3)))
        ax3.set_yticks(np.arange(len(self.tokens3)))
        ax3.set_xticklabels(self.tokens3, rotation=90)
        ax3.set_yticklabels(self.tokens3)
        self.im_diff = ax3.imshow(diff)
        self._text_colorchange(ax3)
        self.cb3 = self.fig.colorbar(self.im_diff, ax=ax3, shrink=0.8, pad=0.1)
        
        # Store original ranges
        self.original_range_im1 = (self.im1.get_array().min(), self.im1.get_array().max())
        self.original_range_im2 = (self.im2.get_array().min(), self.im2.get_array().max())
        self.original_range_imdiff = (self.im_diff.get_array().min(), self.im_diff.get_array().max())

    def _submit_layeridx(self, text):
        """ Handle when user inputs a new layer index """
        #if not self.safe_tamper_flag and len(self.interesting_attns) != 0:
            #print("Error: do not directly change the attention head field and layer field while in a demo. ")
            #print("If you would like to explore the heads on your own, please run attention_visualizer.py." 
                  #"Check usages by running python attention_visualizer.py")
            #plt.close()
            #exit(1)
        # if the text is not a digit, then the number is reverted back to the layer it was before
        old_layer_idx = self.cur_layer_idx  # store previous layer
        if not text.isdigit():
            print(f"Warning: Invalid layer number '{text}'. Reverting to previous layer {old_layer_idx}.")
            self.safe_tamper_flag = True
            self.layer_textbox.set_val(str(old_layer_idx))
            return
        #if the number inputted is a negative or beyond the amount of layers that we have, then the old layer is again put back in place
        new_layer_idx = int(text)
        if new_layer_idx < 0 or new_layer_idx >= self.total_num_layers:
            print(f"Warning: Layer {new_layer_idx} is out of range. Reverting to previous layer {old_layer_idx}.")
            self.safe_tamper_flag = True
            self.layer_textbox.set_val(str(old_layer_idx))
            return
        #if not text.isdigit():
            #print("Error: You entered an invalid layer number. Program exit")
            #plt.close()
            #exit(1)
        self.cur_layer_idx = int(text)
        self._plot_attention_head(self.cur_head_idx, self.cur_layer_idx)
        self.range_slider.reset()
        self.safe_tamper_flag = False

    def _submit_headidx(self, text):
        """ Handle when user inputs a new head index """
        #if not self.safe_tamper_flag and len(self.interesting_attns) != 0:
            #print("Error: do not directly change the attention head field and layer field while in a demo. ")
            #print("If you would like to explore the heads on your own, please run attention_visualizer.py.\n" 
                  #"Check usages by running python attention_visualizer.py")
            #plt.close()
            #exit(1)
        old_head_idx = self.cur_head_idx  # store previous head
        if not text.isdigit():
            print(f"Warning: Invalid head number '{text}'. Reverting to previous head {old_head_idx}.")
            self.safe_tamper_flag = True
            self.head_textbox.set_val(str(old_head_idx))
            return

        new_head_idx = int(text)
        if new_head_idx < 0 or new_head_idx >= self.num_heads_per_layer:
            print(f"Warning: Head {new_head_idx} is out of range. Reverting to previous head {old_head_idx}.")
            self.safe_tamper_flag = True
            self.head_textbox.set_val(str(old_head_idx))
            return
        #if not text.isdigit():
            #print("Error: You entered an invalid layer number. Program exit")
            #plt.close()
            #exit(1)
        self.cur_head_idx = int(text)
        self._plot_attention_head(self.cur_head_idx, self.cur_layer_idx)
        self.range_slider.reset()
        self.safe_tamper_flag = False

    def _init_text_boxes(self):
        self.layer_textbox_ax = self.fig.add_axes([0.45, 0.945, 0.05, 0.05])
        self.head_textbox_ax = self.fig.add_axes([0.55, 0.945, 0.05, 0.05])
        self.layer_textbox = TextBox(self.layer_textbox_ax, label='Layer ', initial=str(self.cur_layer_idx))
        self.head_textbox = TextBox(self.head_textbox_ax, label='Head ', initial=str(self.cur_head_idx))
        self.layer_textbox.on_submit(self._submit_layeridx)
        self.head_textbox.on_submit(self._submit_headidx)
        self.layer_textbox.label.set_fontsize(16)
        self.layer_textbox.text_disp.set_fontsize(16)
        self.head_textbox.label.set_fontsize(16)
        self.head_textbox.text_disp.set_fontsize(16)

    def _init_attn_head_count_label(self):
        self.attention_head_count_label_ax = self.fig.add_axes([0.7, 0.945, 0.05, 0.05])
        self.attention_head_count_label_ax.axis("off")

    def _update_attn_head_count(self):
        self.attention_head_count_label_ax.cla()
        self.attention_head_count_label_ax.axis("off")
        label_text = f"Attention Head {self.cur_overall_idx + 1} of {len(self.interesting_attns)}"
        self.attention_head_count_label_ax.text(0.0, 0.5, label_text, fontsize=12, va="center", ha="left")
        
    def _plot_attention_head(self, head_idx, layer_idx):
        """Plot attention for a specific head."""
        self.cur_head_idx = head_idx
        self.cur_layer_idx = layer_idx
        self.cur_layer_attentions = self.outputs.encoder_attentions[self.cur_layer_idx]

        # Handle colorbar loading
        if self.cb1: self.cb1.remove()
        if self.cb2: self.cb2.remove()
        if self.cb3: self.cb3.remove()

        # initialize attention head count if we're in a demo
        if len(self.interesting_attns) > 0 and self.attention_head_count_label_ax is None:
            self._init_attn_head_count_label()

        # Initialize the textboxes
        if not self.head_textbox and not self.layer_textbox:
            self._init_text_boxes()

        self.fig.subplots_adjust(
            left=0.075, right=0.925,
            top=0.9, bottom=0.1,
            wspace=0.5
        )
        
        # Clear previous visualizations
        for ax in self.axs.flatten():
            ax.clear()
            
        # Reinitialize visualizations
        self._init_matrix_visualizations(self.axs[0, 0], self.axs[0, 1], self.axs[0, 2])
        self._init_line_visualizations(self.axs[1, 0], self.axs[1, 1], self.axs[1, 2])

        self._init_tooltip()

        # Update the attention head count if we're in a demo
        if len(self.interesting_attns) > 0:
            self._update_attn_head_count()
        
        self.fig.canvas.draw()
        
    def _reset_lines(self):
        """Reset line highlighting."""
        # for line in self.hovered_lines:
        #     line.set_alpha(0.3)
        # self.hovered_lines.clear()
        for line in self.all_lines1:
            line.set_visible(True)
        for line in self.all_lines2:
            line.set_visible(True)
        for line in self.all_lines3:
            line.set_visible(True)
        for line in self.hovered_lines:
            line.remove()
        self.hovered_lines.clear()
        self.fig.canvas.draw_idle()

    def _click_line_visualizations(self, event):
        hovered_ax = event.inaxes
        if hovered_ax not in self.axs:
            self._reset_lines()
            return
        # check if the hovered axes are correct, handle accordingly
        if hovered_ax == self.axs[1, 0]:
            attentions = self.cur_layer_attentions[0, self.cur_head_idx, :, :].numpy()
            tokens = self.tokens1
        elif hovered_ax == self.axs[1, 1]:
            attentions = self.cur_layer_attentions[1, self.cur_head_idx, :, :].numpy()
            tokens = self.tokens2
        elif hovered_ax == self.axs[1, 2]:
            attentions = (self.cur_layer_attentions[0, self.cur_head_idx, :, :].numpy()
                          - self.cur_layer_attentions[1, self.cur_head_idx, :,:].numpy())
            tokens = self.tokens3
        else:
            self._reset_lines()
            return

        token_bounds = self._compute_tokenbounds(tokens, spacing=1 / len(tokens))
        x, y = event.x, event.y
        inv = hovered_ax.transAxes.inverted()
        x_axes, y_axes = inv.transform((x, y))

        hovered_token = None
        for i, (ymin, ymax) in enumerate(token_bounds):
            if ymin <= y_axes <= ymax:
                hovered_token = i
                break

        if hovered_token is None: return

        if hovered_ax == self.axs[1, 0]:
            for line in self.all_lines1:
                line.set_visible(False)
        elif hovered_ax == self.axs[1, 1]:
            for line in self.all_lines2:
                line.set_visible(False)
        elif hovered_ax == self.axs[1, 2]:
            for line in self.all_lines3:
                line.set_visible(False)

        if self.hovered_lines:
            for line in self.hovered_lines:
                line.remove()
            self.hovered_lines.clear()

        if x_axes < 0.5:
            for j in range(len(tokens)):
                if abs(attentions[hovered_token, j]) > 0.01:
                    if hovered_ax == self.axs[1, 0]:
                        new_line = self._draw_line_prompts(hovered_ax, hovered_token, j, attentions[hovered_token, j],
                                                     spacing=1 / len(tokens), color="blue")
                    elif hovered_ax == self.axs[1, 1]:
                        new_line = self._draw_line_prompts(hovered_ax, hovered_token, j, attentions[hovered_token, j],
                                                     spacing=1 / len(tokens), color="blue")
                    if hovered_ax == self.axs[1, 2]:
                        new_line = self._draw_line_diff(hovered_ax, hovered_token, j, attentions[hovered_token, j],
                                                  spacing=1 / len(tokens))
                    self.hovered_lines.append(new_line)
        else:
            for i in range(len(tokens)):
                if abs(attentions[i, hovered_token]) > 0.01:
                    if hovered_ax == self.axs[1, 0]:
                        new_line = self._draw_line_prompts(hovered_ax, i, hovered_token, attentions[i, hovered_token],
                                                     spacing=1 / len(tokens), color="blue")
                    elif hovered_ax == self.axs[1, 1]:
                        new_line = self._draw_line_prompts(hovered_ax, i, hovered_token, attentions[i, hovered_token],
                                                     spacing=1 / len(tokens), color="blue")
                    if hovered_ax == self.axs[1, 2]:
                        new_line = self._draw_line_diff(hovered_ax, i, hovered_token, attentions[i, hovered_token],
                                                  spacing=1 / len(tokens))
                    self.hovered_lines.append(new_line)
        self.fig.canvas.draw_idle()
        
    def _on_hover(self, event):
        """Handle mouse hover events over the matrices and line visualizations"""
        # print("All axes:", [ax for ax in self.axs.flatten()])
        # print("event.inaxes:", event.inaxes)

        if event.inaxes is None:
            # print(f"event.inaxes: {event.inaxes}")
            for tooltip in self.tooltips.values():
                if tooltip.get_visible():
                    tooltip.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        hovered_ax = event.inaxes
        if hovered_ax not in self.axs.flatten():
            return

        x_pos = round(event.xdata)
        y_pos = round(event.ydata)

        attentions = None
        tokens_x = None
        tokens_y = None
        if hovered_ax == self.axs[0, 0]:
            attentions = self.cur_layer_attentions[0, self.cur_head_idx, :, :].numpy()
            tokens_x = self.tokens1
            tokens_y = self.tokens1
        elif hovered_ax == self.axs[0, 1]:
            attentions = self.cur_layer_attentions[1, self.cur_head_idx, :, :].numpy()
            tokens_x = self.tokens2
            tokens_y = self.tokens2
        elif hovered_ax == self.axs[0, 2]:
            attentions = (self.cur_layer_attentions[0, self.cur_head_idx, :, :].numpy()
                          - self.cur_layer_attentions[1, self.cur_head_idx, :, :].numpy())
            tokens_x = self.tokens3
            tokens_y = self.tokens3
        else:
            self._click_line_visualizations(event)
            return

        tooltip = self.tooltips[hovered_ax]

        if 0 <= x_pos < attentions.shape[1] and 0 <= y_pos < attentions.shape[0]:  # shape[0] is num rows, shape[1] is num cols
            cur_attention = attentions[y_pos, x_pos]
            tooltip.xy = (x_pos, y_pos)
            tooltip.set_text(f"input: {tokens_y[y_pos]}\noutput: {tokens_x[x_pos]}\nactivation: {cur_attention:.3f}")
            tooltip.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            tooltip.set_visible(False)

        for ax, other_tooltip in self.tooltips.items():
            if ax != hovered_ax and other_tooltip.get_visible():
                other_tooltip.set_visible(False)

        self.fig.canvas.draw_idle()
            
    def _on_unhover(self, event):
        """Handle mouse unhover events."""
        for tooltip in self.tooltips.values():
            tooltip.set_visible(False)
        self._reset_lines()
        self.fig.canvas.draw_idle()
        
    def _next_attention_head(self, event):
        """Move between attention heads and layers.

        - Left/right arrows: navigate 'interesting' heads (demo mode) or all heads (base mode)
        - Up/down arrows: always navigate through every layer/head combination
        - 'q': quit
        """
        if event.inaxes in [self.layer_textbox_ax, self.head_textbox_ax]:
            return

        # UP/DOWN: always full traversal through all heads/layers
        if event.key in ['up', 'down']:
            if event.key == 'up':
                # move forward through all heads/layers
                if self.cur_layer_idx == self.total_num_layers - 1 and self.cur_head_idx == self.num_heads_per_layer - 1:
                    self.cur_layer_idx = 0
                    self.cur_head_idx = 0
                elif self.cur_head_idx == self.num_heads_per_layer - 1:
                    self.cur_head_idx = 0
                    self.cur_layer_idx += 1
                else:
                    self.cur_head_idx += 1
            elif event.key == 'down':
                # move backward through all heads/layers
                if self.cur_layer_idx == 0 and self.cur_head_idx == 0:
                    self.cur_layer_idx = self.total_num_layers - 1
                    self.cur_head_idx = self.num_heads_per_layer - 1
                elif self.cur_head_idx == 0:
                    self.cur_head_idx = self.num_heads_per_layer - 1
                    self.cur_layer_idx -= 1
                else:
                    self.cur_head_idx -= 1

            self.safe_tamper_flag = True
            self.layer_textbox.set_val(str(self.cur_layer_idx))
            self.safe_tamper_flag = True
            self.head_textbox.set_val(str(self.cur_head_idx))
            return

        # LEFT/RIGHT: demo behavior (interesting heads) or all if base mode
        if len(self.interesting_attns) == 0:
            # base mode — left/right also scroll through all heads/layers
            if event.key == 'right':
                if self.cur_layer_idx == self.total_num_layers - 1 and self.cur_head_idx == self.num_heads_per_layer - 1:
                    self.cur_layer_idx = 0
                    self.cur_head_idx = 0
                elif self.cur_head_idx == self.num_heads_per_layer - 1:
                    self.cur_head_idx = 0
                    self.cur_layer_idx += 1
                else:
                    self.cur_head_idx += 1
            elif event.key == 'left':
                if self.cur_layer_idx == 0 and self.cur_head_idx == 0:
                    self.cur_layer_idx = self.total_num_layers - 1
                    self.cur_head_idx = self.num_heads_per_layer - 1
                elif self.cur_head_idx == 0:
                    self.cur_head_idx = self.num_heads_per_layer - 1
                    self.cur_layer_idx -= 1
                else:
                    self.cur_head_idx -= 1
        else:
            # demo mode — left/right scroll through only interesting heads
            if event.key == 'right':
                self.cur_overall_idx = (self.cur_overall_idx + 1) % len(self.interesting_attns)
            elif event.key == 'left':
                self.cur_overall_idx = (self.cur_overall_idx - 1) % len(self.interesting_attns)
            else:
                return
            self.cur_layer_idx, self.cur_head_idx = self.interesting_attns[self.cur_overall_idx]

        # Quit option
        if event.key == 'q':
            print("Exited from the demonstration.")
            plt.close(self.fig)
            return

        # Update UI
        self.safe_tamper_flag = True
        self.layer_textbox.set_val(str(self.cur_layer_idx))
        self.safe_tamper_flag = True
        self.head_textbox.set_val(str(self.cur_head_idx))

        
    def visualize(self):
        """Create and display the attention visualization."""
        # Create figure and subplots
        self.fig, self.axs = plt.subplots(2, 3, figsize=UI_CONFIG['figure_size'])
        self.fig.subplots_adjust(
            left=0.075, right=0.925,
            top=0.9, bottom=0.1,
            wspace=0.5
        )
        
        # Initialize UI elements
        self._init_slider()
        self._init_text_boxes()
        
        # Initial plot
        self._plot_attention_head(self.cur_head_idx, self.cur_layer_idx)
        
        # Connect events
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.fig.canvas.mpl_connect('axes_leave_event', self._on_unhover)
        self.fig.canvas.mpl_connect('key_press_event', self._next_attention_head)
        
        # plt.tight_layout()
        plt.show()


def main():
    """Main function for attention visualization."""
    prompt1, prompt2, demo_type = "", "", None
    if len(sys.argv) < 3:
        print("Usage: python attention_visualizer.py <prompt1> <prompt2> <demo_type>")
        print("<demo_type>: supports 'base', 'pronoun_resolution', 'number_agreement', 'noun_phrases', and 'prep_phrase_attach'")
        print("      base: All attention heads in all layers are shown. Intended for exploration purposes")
        print("      pronoun_resolution: Attention heads of interest exhibiting pronoun resolution are shown")
        print("      number_agreement: Attention heads of interest exhibiting number agreement are shown")
        print("      noun_phrases: Attention heads of interest exhibiting noun phrase identification are shown")
        print("      prep_phrase_attachment: Attention heads of interest exhibiting prepositional phrase attachment are shown")
        print("Note that for the latter four demo_type options, it would make the most sense to input prompts that demonstrate"
              "the option specified.\n")
        print("Example Usage: \n"
              "python attention_visualizer.py 'The man gave the woman his jacket.' 'The man gave the woman her jacket.' 'pronoun_resolution'" 
              "\n\n")
        sys.exit(1)
    elif len(sys.argv) == 3:
        prompt1, prompt2 = sys.argv[1], sys.argv[2]
        demo_type = 'base'
    elif len(sys.argv) == 4:
        prompt1, prompt2 = sys.argv[1], sys.argv[2]
        demo_type = sys.argv[3]
    
    # Initialize model and visualizer
    model_manager = ModelManager("google/flan-t5-large")
    visualizer = AttentionVisualizer(model_manager, [prompt1, prompt2], demo_type)
    
    # Display visualization
    visualizer.visualize()


if __name__ == '__main__':
    main()  