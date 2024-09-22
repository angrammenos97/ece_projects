module fifo_property #(DEPTH=16)
											(clk, rst,fifo_write,fifo_read,fifo_full,fifo_empty,wr_ptr,rd_ptr,cnt);
input logic [7:0] wr_ptr, rd_ptr, cnt;
input logic clk, rst, fifo_write, fifo_read, fifo_full, fifo_empty;

always_comb begin: b1
	if (!rst)
		reset_check: assert final (!rd_ptr && !wr_ptr && fifo_empty && !fifo_full) else $error($stime,,,"FAIL: not reset");
	else
	begin
		if (cnt == 0)
			empty_check: assert final (fifo_empty) else $error($stime,,,"FAIL: not emptied");
		else if (cnt == DEPTH)
			full_check: assert final (fifo_full) else $error($stime,,,"FAIL: not fulled");
	end
end

default disable iff (!rst);
default clocking posedgeClock @(posedge clk); endclocking

property full_prop;
	(fifo_full && fifo_write && !fifo_read) |=> $stable(wr_ptr);
endproperty
write_full_assert: assert property(full_prop) else $error($stime,,,"FAIL: wrote on full");
write_full_cover: cover property(full_prop) $display($stime,,,"PASS: not wrote on full");

property empty_prop;
	(fifo_empty && fifo_read && !fifo_write) |=> $stable(rd_ptr);
endproperty
read_empty_assert: assert property(empty_prop) else $error($stime,,,"FAIL: read on empty");
read_empty_cover: cover property(empty_prop) $display($stime,,,"PASS: not read on empty");

always @(posedge clk, rst)
	$display($stime,,,"rst=%b|fifo_write=%b|fifo_read=%b|wr_prt=%u|rd_ptr=%u|cnt=%u|fifo_full=%b|fifo_empty=%b",
										rst,fifo_write,fifo_read,wr_ptr,rd_ptr,cnt,fifo_full,fifo_empty);

endmodule
