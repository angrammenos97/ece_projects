module fifo #(parameter WIDTH=16, DEPTH=16)
						(fifo_data_in,rst, fifo_write,fifo_read,clk,fifo_data_out,fifo_full,fifo_empty);
input logic [WIDTH-1:0] fifo_data_in;
input logic rst, fifo_write,fifo_read,clk;
output logic [WIDTH-1:0] fifo_data_out;
output logic fifo_full,fifo_empty;

logic [7:0] wr_ptr, rd_ptr, cnt;
logic [DEPTH-1:0][WIDTH-1:0] data;

always_comb
begin
	if(cnt == DEPTH)
	begin
		fifo_full = 1;
		fifo_empty = 0;
	end
	else if (cnt > 0)
	begin
		fifo_full = 0;
		fifo_empty = 0;
	end
	else
	begin
		fifo_full = 0;
		fifo_empty = 1;
	end
end

always_ff @(posedge clk, negedge rst)
begin
	if(!rst)
	begin
		wr_ptr = 0;
		rd_ptr = 0;
		cnt = 0;
	end
	else if (fifo_write)
	begin
		if (!fifo_full)
		begin
			data[wr_ptr] = fifo_data_in;
			wr_ptr = wr_ptr < DEPTH-1 ? wr_ptr+1 : 0;
			cnt++;
		end
	end
	else if (fifo_read)
	begin
		if (!fifo_empty)
		begin
			fifo_data_out = data[rd_ptr];
			rd_ptr = rd_ptr < DEPTH-1 ? rd_ptr+1 : 0;
			cnt--;
		end
	end
end

endmodule
