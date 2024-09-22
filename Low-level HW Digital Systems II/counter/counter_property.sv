module counter_property(data_out, data_in, rst, ld_cnt, updn_cnt, count_enb, clk);
input logic [15:0] data_out, data_in;
input logic rst, ld_cnt, updn_cnt, count_enb, clk;

always_comb begin: b1
	if(!rst)
		reset_check: assert final (data_out==0) else $error($stime,,,"FAIL: not reset");
end

default disable iff (!rst)
default clocking posedgeClock @(posedge clk); endclocking

property load_prop;
	(!ld_cnt) |=> (data_out == data_in);
endproperty
load_check_assert: assert property(load_prop) else $error($stime,,,"FAIL: not loaded");
load_check_cover: cover property(load_prop) $display($stime,,,"PASS: loaded");

property nocount_prop;
	(!count_enb) |=> $stable(data_out);
endproperty
stop_check_assert: assert property(nocount_prop) else $error($stime,,,"FAIL: not stopped");
stop_check_cover: cover property(nocount_prop) $display($stime,,,"PASS: stopped");

property upcount_prob;
	(ld_cnt && count_enb && updn_cnt) |=> (data_out == $past(data_out)+1);
endproperty
up_check_assert: assert property(upcount_prob) else $error($stime,,,"FAIL: not up counted");
up_check_cover: cover property(upcount_prob) $display($stime,,,"PASS: up counted");

property dncount_prob;
	(ld_cnt && count_enb && !updn_cnt) |=> (data_out == $past(data_out)-1);
endproperty
dn_check_assert: assert property(dncount_prob) else $error($stime,,,"FAIL: not down counted");
dn_check_cover: cover property(dncount_prob) $display($stime,,,"PASS: down counted");

always @(posedge clk, rst)
	$display($stime,,,"rst=%b|ld_cnt=%b|count_enb=%b|updn_cnt=%b|data_in=%d|data_out=%d|past=%d",rst,ld_cnt,count_enb,updn_cnt,data_in,data_out,$past(data_out));

endmodule
