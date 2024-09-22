module counter(data_out, data_in, rst, ld_cnt, updn_cnt, count_enb, clk);
output logic [15:0] data_out;
input logic [15:0] data_in;
input logic rst, ld_cnt, updn_cnt, count_enb, clk;

always_ff @(posedge clk, negedge rst)
begin
	if (!rst)
		data_out = 0;
	else if (!ld_cnt)
		data_out = data_in;
	else if (count_enb)
	begin
		if(updn_cnt)
			data_out += 1;
		else
			data_out -= 1;
	end
end

endmodule